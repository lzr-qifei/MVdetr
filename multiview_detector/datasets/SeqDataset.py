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
from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt
from kornia.geometry.transform import warp_perspective
from torch.utils.data import DataLoader
import random
from math import floor
from random import randint
def build_mv_dataset(config):
    from datasets.MultiviewX import MultiviewX
    from datasets.Wildtrack import Wildtrack
    data_path = os.path.join(config['DATA_ROOT'],config['DATASET'][0])
    if 'Wildtrack' in config['DATASET']:
        base = Wildtrack(data_path)
    elif 'MultiviewX' in config['DATASET']:
        # base = MultiviewX(os.path.expanduser('./Data/MultiviewX'))
        base = MultiviewX(data_path)
    return SeqDataset(base,config)

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
class frameDataset(VisionDataset):
    def __init__(self, base, train=True, reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True,
                 semi_supervised=0.0, dropout=0.0, augmentation=False):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.world_from_img, self.img_from_world = self.get_world_imgs_trans()
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        # self.imgs_region = kornia.warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest',
        #                                            align_corners=False)
        self.imgs_region = warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest',
                                                   align_corners=False)
        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.keeps = {}
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        num_keep, num_all = 0, 0
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                keep = np.mean(np.array(frame_range) < frame) < self.semi_supervised if self.semi_supervised else 1
                with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                if keep:
                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        # print('xy: ',grid_x,grid_y)
                        if pedestrian['personID'] not in self.pid_dict:
                            self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                        num_all += 1
                        num_keep += keep
                        num_world_bbox += keep
                        if self.base.indexing == 'xy':
                            world_pts.append((grid_x, grid_y))
                        else:
                            world_pts.append((grid_y, grid_x))
                        world_pids.append(self.pid_dict[pedestrian['personID']])
                        for cam in range(self.num_cam):
                            if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                                img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                      (pedestrian['views'][cam]))
                                img_pids[cam].append(self.pid_dict[pedestrian['personID']])
                                num_imgs_bbox += 1
                self.world_gt[frame] = (np.array(world_pts), np.array(world_pids))
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (np.array(img_bboxs[cam]), np.array(img_pids[cam]))
                self.keeps[frame] = keep

        print(f'all: pid: {len(self.pid_dict)}, frame: {num_frame}, keep ratio: {num_keep / num_all:.3f}\n'
              f'recorded: world bbox: {num_world_bbox / num_frame:.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / num_frame / self.num_cam:.1f}')
        # gt in mot format for evaluation
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()
        # self.prepare_gt_with_id()
        pass

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        # world grid change to xy indexing
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                       self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam],
                                                                          z / self.base.worldcoord_unit)
                                         for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                     for cam in range(self.num_cam)]
        world_from_img = torch.tensor(np.stack(proj_mats))
        # img(xy)_from_worldgrid(xy)
        img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
        return world_from_img.float(), img_from_world.float()

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def prepare_gt_with_id(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                id = int(single_pedestrian['personID'])
                og_gt.append(np.array([frame, grid_x, grid_y,id]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        path = self.gt_fpath.split('.txt')[0]+'_id'+'.txt'
        np.savetxt(path, og_gt, '%d')


    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.world_gt.keys())[index]
        # imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        # inverse_M = torch.inverse(
        #     torch.cat([affine_mats, torch.tensor([0, 0, 1]).view(1, 1, 3).repeat(self.num_cam, 1, 1)], dim=1))[:, :2]
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        # imgs_gt['heatmap_mask'] = self.imgs_region if self.keeps[frame] else torch.zeros_like(self.imgs_region)
        # imgs_gt['heatmap_mask'] = kornia.warp_perspective(imgs_gt['heatmap_mask'], affine_mats, self.img_shape,
        #                                                   align_corners=False)
        # imgs_gt['heatmap_mask'] = F.interpolate(imgs_gt['heatmap_mask'], self.Rimg_shape, mode='bilinear',
        #                                         align_corners=False).bool().float()
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            drop_cam = np.random.randint(0, self.num_cam)
            keep_cams[drop_cam] = 0
            for key in imgs_gt:
                imgs_gt[key][drop_cam] = 0
        # world gt
        world_pt_s, world_pid_s = self.world_gt[frame]
        # print('world_pt_s: ',world_pt_s)
        
        # world_gt = {'world_pts':torch.from_numpy(world_pt_s),'world_pids':torch.from_numpy(world_pid_s),
        #             'world_labels':torch.from_numpy(world_labels)}
        world_gt = get_world_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        return imgs, world_gt, imgs_gt, affine_mats, frame

    def __len__(self):
        return len(self.world_gt.keys())
from torch.utils.data import Dataset
class SeqDataset(Dataset):
    def __init__(self,base,config:dict):
        self.base = base
        self.dataset_name = config['DATASET'][0]
        if self.dataset_name == 'MultiviewX':
            self.num_cam = 6
        self.data_root = config['DATA_ROOT']
        self.dataset = self.get_dataset_structure(dataset=config['DATASET'][0])
        self.infos = self.get_dataset_infos()
        print('check: ',self.infos['MultiviewX'][18][1])
        
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
        self.sample_frames_begin = []
        # self.train = is_train
        self.length = config['LENGTH_PER_SEQUENCE']
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
        frames_idx = self.sample_frames_idx(dataset=dataset, split=split, seq=seq, begin=begin)
        # dataset = self.dataset_name
        images, affine_mats,infos = self.get_multi_frames(dataset=dataset, split=split, seq=seq, frames=frames_idx)

        return {
            # "images": stacked_images,
            "images": images,
            "infos": infos,
            "affine_mats": affine_mats
        }
    
    def get_dataset_structure(self, dataset: str):
        dataset_dir = os.path.join(self.data_root, dataset)
        structure = {"dataset": dataset}
        if dataset =='MultiviewX':
            split = 'Image_subsets'
            split_dir = os.path.join(dataset_dir,split)
            self.cam_names = os.listdir(split_dir)
            seqs = range(36)
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
        gt_path = os.path.join(self.data_root,dataset_name,'train','gt_id.txt')
        if gt_path is not None:
            for f in range(360):
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
                        # [frame, id, x, y, w, h, 1, 1, 1]
                        # f, x, y, id = line.split(" ")
                        # a = line.strip().strip('[]').split(' ')
                        # print(a)
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
            else:                               # real video, do something to sample:
                remain_frames = int(max(self.infos[dataset][seq].keys())) - begin
                max_interval = floor(remain_frames / (self.sample_length - 1))
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
            
            for cam in self.cam_names:
                fid = f"{frame:04}"
                # path = os.path.join(self.dataset['cams'][cam]['image_path'],fid,'.png')
                # print(path)
                image = np.array(Image.open(os.path.join(self.dataset['cams'][cam]['image_path'],fid+'.png')).convert('RGB'))
                image = self.transform(image)
                images.append(image)
                M = np.eye(3)
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
        info["ori_width"], info["ori_height"] = self.img_shape
        pts, ids, labels = list(), list(), list()
        
        try:
            for _,id,label,x,y in self.infos[dataset][seq][frame]["gts"]:
                pts.append([x,y])
                # print(pts)
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

        # image = Image.open(self.infos[dataset][split][seq][frame]["image_path"])
        # info = dict()
        # # Details about current image:
        # info["image_path"] = self.infos[dataset][split][seq][frame]["image_path"]
        # info["dataset"] = dataset
        # info["split"] = split
        # info["seq"] = seq
        # info["frame"] = frame
        # info["ori_width"], info["ori_height"] = image.size
        # # GTs for current image:
        # boxes, ids, labels, areas = list(), list(), list(), list()
        # for _, i, label, _, x, y, w, h in self.infos[dataset][split][seq][frame]["gts"]:
        #     boxes.append([x, y, w, h])
        #     areas.append(w * h)
        #     ids.append(i)
        #     labels.append(label)
        # assert len(boxes) == len(areas) == len(ids) == len(labels), f"GT for [{dataset}][{split}][{seq}][{frame}], " \
        #                                                             f"different attributes have different length."
        # assert len(boxes) > 0, f"GT for [{dataset}][{split}][{seq}][{frame}] is empty."
        # info["boxes"] = torch.as_tensor(boxes, dtype=torch.float)   # in format [x, y, w, h]
        # info["areas"] = torch.as_tensor(areas, dtype=torch.float)
        # info["ids"] = torch.as_tensor(ids, dtype=torch.long)
        # info["labels"] = torch.as_tensor(labels, dtype=torch.long)
        # # Change boxes' format into [x1, y1, x2, y2]
        # info["boxes"][:, 2:] += info["boxes"][:, :2]

        # return image, info

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




