import os
import argparse
import torch.distributed

from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed,infos_to_detr_targets,batch_iterator
from log_engine.logger import Logger, parser_to_dict
from configs.utils import update_config, load_super_config

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
from multiview_detector.models.criterion import SetCriterion
from multiview_detector.models.matcher import HungarianMatcher
from torch.cuda.amp import GradScaler
from torch import optim
from torch import nn
def main(config: dict):
    train_states = {
        "start_epoch": 0,
        "global_iter": 0
    }
    dataset_train = build_dataset(config=config)
    # dataset_train.set_epoch(0)

    model = MVDeTr_w_dec(args=None,dataset=dataset_train).cuda()
    device = 'cuda:0'
    losses = ['labels','center']
    lr = config['LR']
    weight_decay = config['WEIGHT_DECAY']
    length_detr_train_frames = config['LENGTH_DETR_TRAIN']

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
    # matcher = HungarianMatcher(cost_class=0.2,cost_pts=2)
    matcher = HungarianMatcher(cost_class=2.0,cost_pts=50.0)
    criterion = SetCriterion(1,matcher,weight_dict,losses)
    logger = Logger(config['LOG_DIR'])
    clip_max_norm=config["CLIP_MAX_NORM"]
    for epoch in range(2):
        dataset_train.set_epoch(epoch)
        sampler_train = build_sampler(dataset=dataset_train, shuffle=True)
        dataloader = build_dataloader(
            dataset=dataset_train,
            sampler=sampler_train,
            batch_size=config["BATCH_SIZE"],
            num_workers=config["NUM_WORKERS"]
        )
        train_one_epoch(config,model,logger,dataloader,id_criterion,criterion
                        optimizer,epoch,
                        states,clip_max_norm,length_detr_train_frames,)






def train_one_epoch(config:dict, model,logger,dataloader:DataLoader,id_criterion:nn.Module,detr_criterion,
                    optimizer: torch.optim, epoch: int,
                    states:dict,clip_max_norm:float,length_detr_train_frames:int):
    for idx,batch in enumerate(dataloader):
        print('single mat: ',batch["mats"][0][0].shape)
        print('mats: ',len(batch["mats"][0]))

        frames = batch["nested_tensors"]
        infos = batch["infos"]
        # print(f'infos: {infos}')
        detr_targets = infos_to_detr_targets(infos=infos, device=device)
        B, T = len(batch["images"]), len(batch["images"][0])
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
        #Without Train:
        if T > length_detr_train_frames:
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
                    detr_no_grad_outputs = model(frames=detr_no_grad_frames)
        else:
            detr_no_grad_outputs = None

        for i in range(length_detr_train_frames):
            cur_train_frame = detr_train_frames.tensors[i,:,:,:].unsqueeze(0).to(device=device)
            affinemats = batch["mats"][0][0].unsqueeze(0)
            detr_train_outputs = model(cur_train_frame,affinemats)
            targets = detr_targets[i]
            # print(targets)
            loss_dict,_ = detr_criterion(detr_train_outputs,targets)
            losses = sum(loss_dict[k] for k in loss_dict.keys())
            losses.backward()
            print(f'idx:{idx},loss:{losses}')
        optimizer.step()
        optimizer.zero_grad()
    return 


if __name__ == '__main__':
    opt = parse_option()                    # runtime options, a subset of .yaml config file (dict).
    # cfg = yaml_to_dict(opt.config_path)     # configs from .yaml file, path is set by runtime options.
    cfg = yaml_to_dict('/root/MVdetr/multiview_detector/configs/test_dataset.yaml')
    # if opt.super_config_path is not None:
    #     cfg = load_super_config(cfg, opt.super_config_path)
    # else:
    #     cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Then, update configs by runtime options, using the different runtime setting.
    main(cfg)