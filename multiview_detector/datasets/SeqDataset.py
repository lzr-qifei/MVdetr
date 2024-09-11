import os
import sys
sys.path.append('/root/MVdetr')
import json
import time
from operator import itemgetter
import copy
from collections import defaultdict
import numpy as np
from PIL import Image
import kornia
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from multiview_detector.utils.projection import *
from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine_seq
import matplotlib.pyplot as plt
from kornia.geometry.transform import warp_perspective
from torch.utils.data import DataLoader
import random
from math import floor
from random import randint
def build_mv_dataset(config,train):
    from datasets.MultiviewX import MultiviewX
    from datasets.Wildtrack import Wildtrack
    data_path = os.path.join(config['DATA_ROOT'],config['DATASET'][0])
    if 'Wildtrack' in config['DATASET']:
        base = Wildtrack(data_path)
    elif 'MultiviewX' in config['DATASET']:
        # base = MultiviewX(os.path.expanduser('./Data/MultiviewX'))
        base = MultiviewX(data_path)
    return SeqDataset(base,config,train)

def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    # reg_mask = np.zeros([top_k], dtype=np.bool)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret

def get_world_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    # print(H,W)
    # heatmap = np.zeros([1, H, W], dtype=np.float32)
    top_k = len(x_s)
    # print('gt_cnt: ',top_k)
    ct_ints = np.zeros([top_k, 2], dtype=np.float32)
    # reg_mask = np.zeros([top_k], dtype=np.bool)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    reduce_x = reduce*W
    # print('reduce_x: ',reduce_x)
    reduce_y = reduce*H
    for k in range(len(v_s)):
        # ct = np.array([x_s[k] , y_s[k]], dtype=np.float32)
        ct = np.array([x_s[k] / reduce_x , y_s[k] / reduce_y], dtype=np.float32)
        # print(ct)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            # draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            # ct_ints[k] = ct_int
            ct_ints[k] = ct
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()
    # print('ctints:',ct_ints)
    world_labels = np.ones(len(ct_ints))
    # world_labels = np.zeros(len(ct_ints))
    ret = {'world_pts': torch.from_numpy(ct_ints), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset),'labels':torch.from_numpy(world_labels)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

from torch.utils.data import Dataset
class SeqDataset(Dataset):
    def __init__(self,base,config:dict,training:bool):
        self.training = training
        self.base = base
        self.dataset_name = config['DATASET'][0]
        if self.dataset_name == 'MultiviewX':
            self.num_cam = 6
            self.train_seqs = int(config['TRAIN_RATIO'] * 40)
        self.data_root = config['DATA_ROOT']
        self.dataset = self.get_dataset_structure(dataset=config['DATASET'][0])
        self.infos = self.get_dataset_infos()
        self.aug = config['AUG']
        
        # print('check: ',self.infos['MultiviewX'][18][1])
        
        self.img_shape = base.img_shape # H,W;
        self.img_reduce = config['IMG_REDUCE']
        self.world_reduce = config['WORLD_REDUCE'] 
        self.worldgrid_shape = base.worldgrid_shape
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])

        self.sample_steps = config["SAMPLE_STEPS"]
        self.sample_lengths = config["SAMPLE_LENGTHS"]
        self.sample_modes = config["SAMPLE_MODES"]
        self.sample_intervals = config["SAMPLE_INTERVALS"]
        self.sample_length = None
        self.sample_mode = None
        self.sample_interval = None
        # self.sample_frames_begin = []
        # self.train = is_train
        # self.length = config['LENGTH_PER_SEQUENCE']
        self.sample_frames_begin = []
        # self.infos = sel
        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()
    
    def __len__(self):
        return len(self.sample_frames_begin)
        # return self.length

    def __getitem__(self, item):
        dataset, seq, begin = self.sample_frames_begin[item]
        # print(dataset)
        split = ' '
        img_list = []
        frames_idx = self.sample_frames_idx(dataset=dataset, split=split, seq=seq, begin=begin)
        # dataset = self.dataset_name
        for i in frames_idx:
            img,aff,infsss = self.get_single_frame(dataset=dataset, split=split, seq=seq, frame=i)
            img_list.append(img)
        images, affine_mats,infos = self.get_multi_frames(dataset=dataset, split=split, seq=seq, frames=frames_idx)

        return {
            # "images": stacked_images,
            "images": images,
            "infos": infos,
            "affine_mats": affine_mats,
            "img_list": img_list
        }
    
    def get_dataset_structure(self, dataset: str):
        dataset_dir = os.path.join(self.data_root, dataset)
        structure = {"dataset": dataset}
        if dataset =='MultiviewX':
            split = 'Image_subsets'
            split_dir = os.path.join(dataset_dir,split)
            self.cam_names = os.listdir(split_dir)
            
            if self.training:
                seqs = range(self.train_seqs)
            else:
                seqs = range(self.train_seqs,40)
            structure['cams'] = {
                cam : {
                    'image_path':os.path.join(split_dir,cam),
                    'gt_path':os.path.join(self.data_root,'gt_id.txt')
                }
                for cam in self.cam_names
            }
            structure['seqs'] = {
                seq :{
                    "max_frame" : 10
                }
                for seq in seqs
            }
        return structure
    def get_dataset_infos(self):
        infos = defaultdict((lambda: defaultdict(lambda: defaultdict(dict))))
        dataset = self.dataset
        cams = dataset['cams']
        dataset_name = dataset["dataset"]
        # for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(self.dataloader):
        # if self.train:
        #     split = 'train'
        # else:
        #     split = 'test'
        if self.training:
            gt_path = os.path.join(self.data_root,dataset_name,'train','gt_id.txt')
            if gt_path is not None:
                for f in range(self.train_seqs * dataset['seqs'][1]['max_frame']):
                    seq = f // 10
                    # f_idx = abs(f-(seq*10))
                    infos[dataset_name][seq][f-(seq*10)]["gts"] = list()
        else:
            gt_path = os.path.join(self.data_root,dataset_name,'test','gt_id.txt')
            if gt_path is not None:
                for f in range(self.train_seqs * dataset['seqs'][36]['max_frame'], 40 * dataset['seqs'][36]['max_frame']):
                    seq = f // 10
                    # f_idx = abs(f-(seq*10))
                    infos[dataset_name][seq][f-(seq*10)]["gts"] = list()
        

            # print('check: ',infos[dataset_name][18][1])

            # print(infos[dataset_name][23][2])
            
        with open(gt_path, "r") as gt_file:
            lines = gt_file.readlines()
            # print(len(lines))
            for line in lines:
                # line = line[:-1]
                if dataset_name == "MultiviewX" :
                    f, x, y, id =  line.strip().strip('[]').split(' ')
                    seq = int(f) // 10
                    label = 1
                elif dataset_name == "MOT17" or dataset_name == "MOT17_SPLIT":
                    pass
                else:
                    raise NotImplementedError(f"Can't analysis the gts of dataset '{dataset_name}'.")
                # format, and write into infos
                f, id, label = map(int, (f, id, label))
                x, y = map(float, (x, y))
                # assert v != 0.0, f"Visibility of object '{i}' in frame '{f}' is 0.0."
                infos[dataset_name][seq][f-(seq*10)]["gts"].append([
                    f, id, label, x, y
                ])
                # print(f'seq:{seq},f:{f},id:{id},xy:{x,y}')
                # if int(f) == 5: 
                #     print(infos[dataset_name][seq][f]["gts"])
                # if infos[dataset_name][seq][f] == {}:
                #     print(f'f={f}')

            # print('check: ',infos[dataset_name][18][1])

        return infos
    
    def sample_frames_idx(self, dataset: str, split: str, seq: str, begin: int) :
        if self.sample_mode == "random_interval":
            if dataset in ["CrowdHuman"]:       # static images, repeat is all right:
                return [begin] * self.sample_length
            elif self.sample_length == 1:       # only train detection:
                return [begin]
            else:                     
                # real video, do something to sample:
                try:
                    remain_frames = int(max(self.infos[dataset][seq].keys())) - begin
                    max_interval = floor(remain_frames / (self.sample_length - 1))
                except:
                    max_interval = 1
                interval = min(randint(1, self.sample_interval), max_interval)      # legal interval
                frames_idx = [begin + interval * _ for _ in range(self.sample_length)]

                # if not all([len(self.infos[dataset][seq][_]["gts"]) for _ in frames_idx]):
                    # In the sampling sequence, there is at least a frame's gt is empty, not friendly for training,
                    # make sure all frames have gt:
                for f in frames_idx:
                    if 'gts' in self.infos[dataset][seq][f].keys():
                        frames_idx = [begin + _ for _ in range(self.sample_length)]
                        # print(f'success seq:{seq},f:{f}')
                    else:
                        print(f'seq:{seq},f:{f}')
                        print(self.infos[dataset][seq][f])
        else:
            raise NotImplementedError(f"Do not support sample mode '{self.sample_mode}'.")
        return frames_idx
    

    def get_multi_frames(self, dataset: str, split: str, seq: str, frames):
        return zip(*[self.get_single_frame(dataset=dataset, split=split, seq=seq, frame=frame) for frame in frames])
    
    def get_single_frame(self, dataset: str, split: str, seq: str, frame: int):
        images = []
        affine_mats = []

        if dataset=='MultiviewX':
            # Rh,Rw = self.base.worldgrid_shape
            # Rh,Rw = Rh / self.co
            H,W = self.Rworld_shape
            world_reduce_H = H*self.world_reduce
            world_reduce_W = W*self.world_reduce
            # print(f'r_h: {world_reduce_H}')
            # print(f'r_w: {world_reduce_W}')
            for cam in self.cam_names:
                f = seq*10+frame
                fid = f"{f:04}"
                # path = os.path.join(self.dataset['cams'][cam]['image_path'],fid,'.png')
                # print(path)
                image = np.array(Image.open(os.path.join(self.dataset['cams'][cam]['image_path'],fid+'.png')).convert('RGB'))
                # print((os.path.join(self.dataset['cams'][cam]['image_path'],fid+'.png')))
                if self.aug:
                    image, M = random_affine_seq(image)
                    image = self.transform(image)
                    images.append(image)
                else:
                    M = np.eye(3)
                    image = self.transform(image)
                    images.append(image)
                affine_mats.append(torch.from_numpy(M).float())
            imgs = torch.stack(images)
            affine_mats = torch.stack(affine_mats)
        elif dataset =='WildTrack':
            #TODO:补充WildTrack的数据结构
            pass
        info = dict()
        info["dataset"] = dataset
        seq = int(seq)
        info["seq"] = seq
        info["frame"] = frame
        imgh,imgw = self.img_shape[0],self.img_shape[1]
        info["ori_width"], info["ori_height"] = imgh / self.img_reduce ,imgw / self.img_reduce
        # info["ori_width"], info["ori_height"] = self.img_shape
        pts, ids, labels = list(), list(), list()
        
        try:
            for _,id,label,x,y in self.infos[dataset][seq][frame]["gts"]:
                # print('ori pts: ',x,y)
                x = x / world_reduce_W
                y = y / world_reduce_H
                pts.append([x,y])
                # print('pts: ',pts)
                
                ids.append(id)
                labels.append(label)
        except:
            print(f'seq: {seq},f:{frame}')
        assert len(pts) == len(labels) == len(ids)
        info["pts"] = torch.as_tensor(pts,dtype=torch.float)
        info["ids"] = torch.as_tensor(ids,dtype=torch.long)
        info["labels"] = torch.as_tensor(labels, dtype=torch.long)
        # all([len(self.infos[dataset][seq][_]["gts"]))

        # print(info)
        return imgs,affine_mats,info


    def set_epoch(self,epoch:int):
        self.sample_frames_begin = []
        for _ in range(len(self.sample_steps)):
            if epoch >= self.sample_steps[_]:
                self.sample_mode = self.sample_modes[_]
                self.sample_length = self.sample_lengths[_]
                self.sample_interval = self.sample_intervals[_]
                break
        
        for seq in self.dataset["seqs"]:
            f_min = 0
            f_max = 9
            for f in range(f_min, f_max - (self.sample_length - 1) + 1):
                # if all([len(self.infos[self.dataset["dataset"]][seq][f + _]["gts"]) > 0
                #         for _ in range(self.sample_length)]):
                self.sample_frames_begin.append(
                    (self.dataset["dataset"],seq,f)
                )
        # print(f'len: {len(self.sample_frames_begin)}')
        return




