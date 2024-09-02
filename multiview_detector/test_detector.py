import os
import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed,infos_to_detr_targets,batch_iterator,combine_detr_outputs,resize_detr_outputs
from log_engine.logger import Logger, parser_to_dict
from log_engine.log import Metrics
from configs.utils import update_config, load_super_config
from torch.utils.data import RandomSampler, SequentialSampler
import time
import numpy as np
from evaluation.evaluate import evaluate
# from train_engine import train
# from eval_engine import evaluate
# from submit_engine import submit

def parse_option():
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--semi_supervised', type=float, default=0)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    # parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    # parser.add_argument('--deterministic', type=str2bool, default=False)
    # parser.add_argument('--augmentation', type=str2bool, default=True)

    parser.add_argument('--world_feat', type=str, default='deform_trans_w_dec',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio','deform_trans_w_dec'])
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    parser.add_argument('--data', type=str, default='./Data')

    parser.add_argument('--two_stage', default=False,action='store_true')

    parser.add_argument('--save',type=int,default=5,help='x, every x epochs save ckpt')
    parser.add_argument('--num_q',type=int,default=300,help='num_queries')
    parser.add_argument('--train_ratio',type=float,default=0.9,help='perception of train set, \
                            0.9 means 90 percent of dataset would be used as train set')
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--pth',type=str,default=None)
    parser.add_argument('--out_path',type=str,default='./results/test.txt')
    parser.add_argument('--det_thres',type=float,default=0.65)
    parser.add_argument('--sensecore', action='store_true')
    parser.add_argument('--vis_path', type=str,default='/root/MVdetr/multiview_detector/vis_results/')
    parser.add_argument('--config_path', type=str,default='/root/MVdetr/multiview_detector/configs/test_dataset.yaml')
    args = parser.parse_args()
    return args
from datasets import build_dataset, build_sampler, build_dataloader
from torch.utils.data import DataLoader
from multiview_detector.models.mvdetr_with_decoder import MVDeTr_w_dec
from utils.nested_tensor import nested_tensor_index_select
from einops import rearrange
from multiview_detector.models.criterion import SetCriterion,IDCriterion
from multiview_detector.models.matcher import HungarianMatcher,HungarianMatcher_batch
from multiview_detector.models.tracker import MVPTR
from torch.cuda.amp import GradScaler
from torch import optim
from torch import nn
def main(config: dict):
    metrics = Metrics()
    train_states = {
        "start_epoch": 0,
        "global_iter": 0
    }
    dataset_train = build_dataset(config=config,train=True)
    dataset_test = build_dataset(config=config,train=False)
    # dataset_train.set_epoch(0)

    # model = MVDeTr_w_dec(args=None,dataset=dataset_train).cuda()
    model = MVPTR(config)
    device = 'cuda:0'
    losses = ['labels','center']
    lr = config['LR']
    weight_decay = config['WEIGHT_DECAY']
    length_detr_train_frames = config['LENGTH_DETR_TRAIN']
    device = config['DEVICE']

    param_dicts = [{"params": [p for n, p in model.named_parameters() if 'base' not in n and p.requires_grad], },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": lr , }, ]
    # optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    weight_dict ={
            'labels':torch.tensor(2,dtype=float,device='cuda:0'),
            'center':torch.tensor(50.0,dtype=float,device='cuda:0'),
            # 'loss_ce':torch.tensor(0.1,dtype=float,device='cuda:0'),
            # 'loss_center':torch.tensor(2,dtype=float,device='cuda:0'),
            # 'offset':torch.tensor(1,dtype=float,device='cuda:0')
    }
    optimizer = optim.Adam(param_dicts, lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()
    matcher = HungarianMatcher(cost_class=2,cost_pts=50)
    matcher_batch = HungarianMatcher_batch(cost_class=2.0,cost_pts=50.0)
    criterion = SetCriterion(config['NUM_CLASSES'],matcher,matcher_batch,weight_dict,losses)
    logger = Logger(config['LOG_DIR'])
    clip_max_norm=config["CLIP_MAX_NORM"]
    id_criterion = IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        # gpu_average=config["ID_LOSS_GPU_AVERAGE"]
        gpu_average=None
    )
    states = None
    if config["TRAIN_STAGE"] == "only_detr":  
        only_detr = True
    else: 
        only_detr = False
    for epoch in range(100):
        dataset_train.set_epoch(epoch)
        dataset_test.set_epoch(epoch)
        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        
        if config['RESUME'] is None:
            dataloader = build_dataloader(
            dataset=dataset_train,
            sampler=sampler_train,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"]
            )   
            train_one_epoch(config,model,logger,dataloader,id_criterion,criterion,
                            optimizer,epoch,
                            states,clip_max_norm,length_detr_train_frames,metrics)
            if (epoch+1) % config["SAVE_CHECKPOINT_PER_EPOCH"] ==0:
                    torch.save(model.state_dict(), os.path.join(config['LOG_DIR'], 'MultiviewDetector_{}.pth'.format(epoch)))
            evaluate_one_epoch(config,model,dataset_test,only_detr)
        else:
            model.load_state_dict(torch.load(config['RESUME']))
            #TODO:加入resume训练模式
            model.to(device)
            
            evaluate_one_epoch(config,model,dataset_test,only_detr)
            break
        


def train_one_epoch(config:dict, model,logger,dataloader:DataLoader,id_criterion:nn.Module,detr_criterion,
                    optimizer: torch.optim, epoch: int,
                    states:dict,clip_max_norm:float,length_detr_train_frames:int,metrics):
    model.train()
    optimizer.zero_grad()
    # detr_params = []
    # other_params = []
    weight_dict ={
            'labels':torch.tensor(2,dtype=float,device='cuda:0'),
            'center':torch.tensor(50.0,dtype=float,device='cuda:0')
    }
    # for name, param in model.named_parameters():
    #     if "detr" in name:
    #         detr_params.append(param)
    #     else:
    #         other_params.append(param)
    for idx,batch in enumerate(dataloader):
        # print('single mat: ',batch["mats"][0][0].shape)
        # print('mats: ',len(batch["mats"][0]))
        B, T = len(batch["images"]), len(batch["images"][0])
        device = config['DEVICE']
        # frames = batch["nested_tensors"]
        frames = batch['images']
        infos = batch["infos"]
        # affinemats = batch["mats"][0][0].unsqueeze(0)

        # affinemats_infer = batch["mats"][0][:T-length_detr_train_frames]
        # affinemats_infer = torch.stack(affinemats_infer,dim=0)
        # print(f'infos: {infos}')
        detr_targets = infos_to_detr_targets(infos=infos, device=device)
        
        random_frame_idxs = torch.randperm(T)
        # length_detr_train_frames = 4
        detr_train_frame_idxs = random_frame_idxs[:length_detr_train_frames]
        detr_no_grad_frame_idxs = random_frame_idxs[length_detr_train_frames:]

        detr_train_frames = nested_tensor_index_select(frames, dim=1, index=detr_train_frame_idxs)
        detr_no_grad_frames = nested_tensor_index_select(frames, dim=1, index=detr_no_grad_frame_idxs)

        detr_train_frames.tensors = rearrange(detr_train_frames.tensors, "b t n c h w -> (b t) n c h w")
        detr_train_frames.mask = rearrange(detr_train_frames.mask, "b t n h w -> (b t) n h w")
        detr_no_grad_frames.tensors = rearrange(detr_no_grad_frames.tensors, "b t n c h w -> (b t) n c h w")
        detr_no_grad_frames.mask = rearrange(detr_no_grad_frames.mask, "b t n h w -> (b t) n h w")
        detr_no_grad_frames = detr_no_grad_frames.to(device)
        detr_no_grad_outputs = None
        detr_train_outputs = None
        #Without Train:
        if T > length_detr_train_frames and config["TRAIN_STAGE"] != "only_detr":
            with torch.no_grad():
                if length_detr_train_frames > 0 and len(detr_no_grad_frames) > length_detr_train_frames * 4:
                    # To reduce CUDA memory usage:
                    detr_no_grad_outputs = None
                    # detr_no_grad_adapter_outputs = None
                    for batch_frames in batch_iterator(length_detr_train_frames * 4, detr_no_grad_frames):
                        batch_frames = batch_frames[0]
                        _ = model(frames=batch_frames)
                        if detr_no_grad_outputs is None:
                            detr_no_grad_outputs = _
                        else:
                            detr_no_grad_outputs = combine_detr_outputs(detr_no_grad_outputs, _)
                else:
                    for i in range(T-length_detr_train_frames):
                        affinemats = batch["mats"][0][i].unsqueeze(0)
                        cur_infer_frame = detr_no_grad_frames.tensors[i,:,:,:].unsqueeze(0).to(device=device)
                        detr_infer_outputs = model(cur_infer_frame,affinemats)
                        # detr_infer_out_list.append(detr_infer_outputs)
                        detr_infer_outputs = resize_detr_outputs(detr_infer_outputs)
                        detr_no_grad_outputs = combine_detr_outputs(detr_infer_outputs,detr_no_grad_outputs)
                    # detr_no_grad_outputs = torch.stack(detr_infer_out_list)
                        # detr_no_grad_outputs = model(detr_no_grad_frames.tensors,affinemats)
        else:
            detr_no_grad_outputs = None

        for i in range(length_detr_train_frames):
            affinemats = batch["mats"][0][i].unsqueeze(0)
            cur_train_frame = detr_train_frames.tensors[i].unsqueeze(0).to(device=device)
            
            cur_train_outputs = model(cur_train_frame,affinemats)
            
            detr_train_loss_dict,_ = detr_criterion(cur_train_outputs,detr_targets[i])
            if i == 5:
                print(detr_train_loss_dict)
            losses = sum(detr_train_loss_dict[k]* weight_dict[k] for k in detr_train_loss_dict.keys())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            cur_train_outputs = resize_detr_outputs(cur_train_outputs)
            detr_train_outputs = combine_detr_outputs(detr_train_outputs,cur_train_outputs)
        # targets = torch.stack(detr_targets)
            # print(targets)
        if detr_no_grad_outputs != None: 
            detr_outputs = combine_detr_outputs(detr_train_outputs, detr_no_grad_outputs)

            detr_loss_dict,match_idxs = detr_criterion(detr_outputs,detr_targets)
        if config["TRAIN_STAGE"] == "only_detr":    # only train detr part:
            id_loss = None
        else:
            match_instances = generate_match_instances(
                match_idxs=match_idxs, infos=infos, detr_outputs=detr_outputs
            )
            model.add_random_id_words_to_instances(instances=match_instances[0])
            pred_id_words, gt_id_words = model.forward_train(
                track_history=match_instances,
                traj_drop_ratio=config["TRAJ_DROP_RATIO"],
                traj_switch_ratio=config["TRAJ_SWITCH_RATIO"] if "TRAJ_SWITCH_RATIO" in config else 0.0,
                use_checkpoint=config["SEQ_DECODER_CHECKPOINT"],
            )
            id_loss = id_criterion(pred_id_words, gt_id_words)

            ##单独对id_loss进行反传
            # losses = sum(loss_dict[k] for k in loss_dict.keys())
            # losses.backward()
        print(f'idx:{idx},loss:{losses}')
        # if (i + 1) % config["ACCUMULATE_STEPS"] == 0:
        #     if clip_max_norm > 0:
        #         detr_grad_norm = torch.nn.utils.clip_grad_norm_(detr_params, clip_max_norm)
        #         other_grad_norm = torch.nn.utils.clip_grad_norm_(other_params, clip_max_norm)
        #         metrics["detr_grad_norm"].update(detr_grad_norm.item())
        #         metrics["other_grad_norm"].update(other_grad_norm.item())
        #     else:
        #         pass
        # optimizer.step()
        # optimizer.zero_grad()
        # optimizer.step()
        # optimizer.zero_grad()
    return 
from structures.instances import Instances
def generate_match_instances(match_idxs, infos, detr_outputs):
    match_instances = []
    B, T = len(infos), len(infos[0])
    for b in range(B):
        match_instances.append([])
        for t in range(T):
            flat_idx = b * T + t
            output_idxs, info_idxs = match_idxs[flat_idx]
            instances = Instances(image_size=(0, 0))
            instances.ids = infos[b][t]["ids"][info_idxs]
            instances.gt_pts = infos[b][t]["pts"][info_idxs]
            instances.pred_ct_pts = detr_outputs["pred_ct_pts"][flat_idx][output_idxs]
            instances.outputs = detr_outputs["outputs"][flat_idx][output_idxs]
            match_instances[b].append(instances)
    return match_instances

def evaluate_one_epoch(config: dict,model: nn.Module,dataset,only_detr: bool = False):
    model.eval()
    device = config["DEVICE"]
    res_fpath = config['OUT_DIR']
    det_thres = 0.65
    gt_fpath = config['DETR_GT']
    with torch.no_grad():
        if only_detr :
            dataloader = build_dataloader(
                dataset=dataset,
                sampler=SequentialSampler(dataset),
                batch_size=config["BATCH_SIZE"],
                num_workers=config["NUM_WORKERS"]
            )
            res_list = []
            for idx,batch in enumerate(dataloader):
                frames = batch["nested_tensors"]
                infos = batch["infos"]
                seq = infos[0][0]['seq']
                # affinemats = batch["mats"][0][0].unsqueeze(0)
                B, T = len(batch["images"]), len(batch["images"][0])
                t0 = time.time()
                for i in range(T):
                    frame_id = seq*T + i
                    # output = model()
                    affinemats = batch["mats"][0][i].unsqueeze(0)
                    cur_infer_frame = frames.tensors.squeeze()[i].unsqueeze(0).to(device=device)
                    outputs = model(cur_infer_frame,affinemats)
                    # detr_infer_outputs = resize_detr_outputs(detr_infer_outputs)
                    # detr_no_grad_outputs = combine_detr_outputs(detr_infer_outputs,detr_no_grad_outputs)
                # for item in detr_no_grad_outputs:
                    if res_fpath is not None:
                        grid_xy,out_logits = outputs['pred_ct_pts'],outputs['pred_logits']
                        
                        if dataloader.dataset.base.indexing == 'xy':
                            positions = grid_xy
                        else:
                            positions = grid_xy[:, :, [1, 0]]
                        # print('scores: ',out_logits[:10])
                        scores = out_logits.sigmoid()
                        # print("scores:",scores.shape)
                        top50_values, topk_indexes = torch.topk(scores.view(1, -1), 50, dim=1)
                        print(top50_values)
                        topk_values = [t for t in top50_values[0] if t > det_thres]
                        # print(len(topk_values))
                        if len(topk_values)<35:
                            max_idx = 35
                        else:
                            max_idx = len(topk_values)
                        # topk_indexes=topk_indexes[0][0:max_idx+1]
                        topk_pts_idx=topk_indexes[0][0:max_idx+1]

                        for b in range(B):
                            pos = positions[topk_pts_idx,:].squeeze()
                            pos_cpu = pos.cpu()
                            pos_cpu[:,0] = pos_cpu[:,0]*1000
                            pos_cpu[:,1] = pos_cpu[:,1]*640
                            frame_idx = torch.ones([pos.shape[0], 1])* frame_id
                            res = torch.cat([frame_idx, pos_cpu], dim=1)
                            res_list.append(res)

            t1 = time.time()
            t_epoch = t1 - t0
            if res_fpath is not None:
                res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
                np.savetxt(res_fpath, res_list, '%d')
                recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                        os.path.abspath(gt_fpath),
                                                        dataloader.dataset.base.__name__)
                print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
            else:
                moda = 0
                        # pass
    return 






if __name__ == '__main__':
    opt = parse_option()                    # runtime options, a subset of .yaml config file (dict).
    # cfg = yaml_to_dict(opt.config_path)     # configs from .yaml file, path is set by runtime options.
    cfg = yaml_to_dict('/root/MVdetr/multiview_detector/configs/test_dataset_detr_only.yaml')
    # if opt.super_config_path is not None:
    #     cfg = load_super_config(cfg, opt.super_config_path)
    # else:
    #     cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    main(cfg)