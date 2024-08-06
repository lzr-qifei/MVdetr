import time
import os
import numpy as np
import torch
from torch import nn, topk
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from PIL import Image
from multiview_detector.loss import *
from multiview_detector.evaluation.evaluate import evaluate
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.nms import nms
from multiview_detector.utils.meters import AverageMeter
from multiview_detector.utils.image_utils import add_heatmap_to_image, img_color_denormalize

## Multiview-DETR trainer

class BaseTrainer(object):
    def __init__(self):
        super(BaseTrainer, self).__init__()


class PerspectiveTrainer(BaseTrainer):
    def __init__(self, model, logdir, cls_thres=0.4, alpha=1.0, use_mse=False, id_ratio=0,two_stage=False):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.mse_loss = nn.MSELoss()
        self.focal_loss = FocalLoss()
        self.regress_loss = RegL1Loss()
        self.ce_loss = RegCELoss()
        self.cls_thres = cls_thres
        self.logdir = logdir
        self.denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.alpha = alpha
        self.use_mse = use_mse
        self.id_ratio = id_ratio
        self.two_stage = two_stage

    def train(self, epoch, dataloader,criterion, optimizer, scaler,device, scheduler=None, log_interval=100):
        self.model.train()
        
        criterion.train()
        # losses = 0
        # losses = torch.tensor(0,dtype=float,device='cuda:0',requires_grad=True)
        t0 = time.time()
        t_b = time.time()
        t_forward = 0
        t_backward = 0
        # device = 'cuda:0'
        for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(dataloader):
            # print('w_gt_shape',world_gt['world_pts'].shape)
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.cuda(device=device)
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])##(B,N,...)---->(B*N,...)##
                
            # with autocast():
            # supervised
            # (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats)
            outputs = self.model(data,affine_mats)
            targets = world_gt
            # loss_dict = criterion(outputs,targets)
            loss_dict,_ = criterion(outputs,targets)
            log="[{}/{}] - ".format(batch_idx, len(dataloader))
            for k,v in loss_dict.items():
                log+="ori_loss_{}: {:.4f}, ".format(k,v)
            # q = loss_dict.cpu()
            # print(q)
            weight_dict = criterion.weight_dict
            
            # print('key: ',loss_dict.keys())
            # for k in loss_dict.keys():
            #     if k in weight_dict.keys():
            #         # print(k)
            #         tmp = loss_dict[k] * weight_dict[k]
            #         # tmp = loss_dict[k]
            #         losses+=tmp
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k]=loss_dict[k] * weight_dict[k]
            for k,v in loss_dict.items():
                log+="weighted_loss_{}: {:.4f}, ".format(k,v)
            # loss_dict={loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}
            losses = sum(loss_dict[k] for k in loss_dict.keys())
            log+="total_loss: {:.4f}".format(losses)
            if batch_idx%10==0:
                print(log)

            #TODO:写ct的loss
            # loss_w_hm = self.focal_loss(world_heatmap, world_gt['heatmap'])
            # loss_w_off = self.regress_loss(world_offset, world_gt['reg_mask'], world_gt['idx'], world_gt['offset'])
            # # loss_w_id = self.ce_loss(world_id, world_gt['reg_mask'], world_gt['idx'], world_gt['pid'])
            # loss_img_hm = self.focal_loss(imgs_heatmap, imgs_gt['heatmap'])
            # loss_img_off = self.regress_loss(imgs_offset, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['offset'])
            # loss_img_wh = self.regress_loss(imgs_wh, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['wh'])
            # loss_img_id = self.ce_loss(imgs_id, imgs_gt['reg_mask'], imgs_gt['idx'], imgs_gt['pid'])

            # multiview regularization

            # w_loss = loss_w_hm + loss_w_off  # + self.id_ratio * loss_w_id
            # img_loss = loss_img_hm + loss_img_off + loss_img_wh * 0.1  # + self.id_ratio * loss_img_id
            # loss = w_loss + img_loss / N * self.alpha
            # if self.use_mse:
            #     loss = self.mse_loss(world_heatmap, world_gt['heatmap'].to(world_heatmap.device)) + \
            #            self.alpha * self.mse_loss(imgs_heatmap, imgs_gt['heatmap'].to(imgs_heatmap.device))

            t_f = time.time()
            t_forward += t_f - t_b

            # scaler.scale(losses).backward()
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.zero_grad()
            grad_total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            losses.backward()
            optimizer.step()

            # losses_total += losses.item()

            t_b = time.time()
            t_backward += t_b - t_f

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
                elif isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts) or \
                        isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR):
                    scheduler.step(epoch - 1 + batch_idx / len(dataloader))
            if (batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(dataloader):
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{(batch_idx + 1)}, loss: {losses / (batch_idx + 1):.6f}, '
                      f'Time: {t_epoch:.1f}')
                pass
            losses_cpu =losses.cpu() 
        return losses_cpu / len(dataloader)

    def test(self, epoch, dataloader, criterion,res_fpath=None, visualize=False,det_thres = 0.55):
        self.model.eval()
        # criterion.eval()
        losses = 0
        res_list = []
        t0 = time.time()
        for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(dataloader):
            B, N = imgs_gt['heatmap'].shape[:2]
            data = data.cuda()
            for key in imgs_gt.keys():
                imgs_gt[key] = imgs_gt[key].view([B * N] + list(imgs_gt[key].shape)[2:])
            # with autocast():
            with torch.no_grad():
                # (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh) = self.model(data, affine_mats)
                outputs = self.model(data,affine_mats,visualize=True)
                # print('logits: ',outputs['pred_logits'])
                targets = world_gt
                # loss_dict = criterion(outputs,targets)
                loss_dict,indices = criterion(outputs,targets)
                # topk_pts_idx = indices[0][0].unsqueeze(0)
                # print(loss_dict)
                weight_dict = criterion.weight_dict
                loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                # print('loss:',loss)



            losses += loss

            if res_fpath is not None:
                # xys = mvdet_decode(torch.sigmoid(world_heatmap.detach().cpu()), world_offset.detach().cpu(),
                #                    reduce=dataloader.dataset.world_reduce)
                # xys = mvdet_decode(world_heatmap.detach().cpu(), reduce=dataloader.dataset.world_reduce)
                # grid_xy, scores = xys[:, :, :2], xys[:, :, 2:3]
                grid_xy,out_logits = outputs['pred_ct_pts'],outputs['pred_logits']
                if dataloader.dataset.base.indexing == 'xy':
                    positions = grid_xy
                else:
                    positions = grid_xy[:, :, [1, 0]]
                # print('scores: ',out_logits[:10])
                scores = out_logits.sigmoid()
                # print("scores:",scores.shape)
                top50_values, topk_indexes = torch.topk(scores.view(1, -1), 50, dim=1)
                # top30_values,_= torch.topk(scores.view(1, -1), 30, dim=1)
                # print('top30 score: ',top30_values)
                # topk_pts_idx = topk_indexes // out_logits.shape[-1]
                # print(top50_values)
                topk_values = [t for t in top50_values[0] if t > det_thres]
                print(len(topk_values))
                if len(topk_values)<35:
                    max_idx = 35
                else:
                    max_idx = len(topk_values)
                # topk_indexes=topk_indexes[0][0:max_idx+1]
                topk_pts_idx=topk_indexes[0][0:max_idx+1]

                for b in range(B):

                    pos = positions[topk_pts_idx,:].squeeze()
                    # print('pos_s shape: ',pos.shape)
                    pos_cpu = pos.cpu()
                    # print(pos_cpu[:,0])
                    pos_cpu[:,0] = pos_cpu[:,0]*1000
                    # print(pos_cpu[:,0])
                    pos_cpu[:,1] = pos_cpu[:,1]*640
                    # print('pos shape:',pos.shape)
                    frame_idx = torch.ones([pos.shape[0], 1])* frame[b]
                    # print('frame_idx',frame_idx.shape)
                    res = torch.cat([frame_idx, pos_cpu], dim=1)
                    # ids, count = nms(pos, s, 20, np.inf)
                    # ids_cpu = ids.cpu()
                    # res = torch.cat([torch.ones([count, 1]) * frame[b], pos_cpu[ids_cpu[:count]]], dim=1)
                    # print(res)
                    res_list.append(res)

        t1 = time.time()
        t_epoch = t1 - t0

        # if visualize:
        #     # visualizing the heatmap for world
        #     fig = plt.figure()
        #     subplt0 = fig.add_subplot(211, title="output")
        #     subplt1 = fig.add_subplot(212, title="target")
        #     subplt0.imshow(world_heatmap.cpu().detach().numpy().squeeze())
        #     subplt1.imshow(world_gt['heatmap'].squeeze())
        #     plt.savefig(os.path.join(self.logdir, f'world{epoch if epoch else ""}.jpg'))
        #     plt.close(fig)
        #     # visualizing the heatmap for per-view estimation
        #     heatmap0_foot = imgs_heatmap[0].detach().cpu().numpy().squeeze()
        #     img0 = self.denormalize(data[0, 0]).cpu().numpy().squeeze().transpose([1, 2, 0])
        #     img0 = Image.fromarray((img0 * 255).astype('uint8'))
        #     foot_cam_result = add_heatmap_to_image(heatmap0_foot, img0)
        #     foot_cam_result.save(os.path.join(self.logdir, 'cam1_foot.jpg'))

        if res_fpath is not None:
            res_list = torch.cat(res_list, dim=0).numpy() if res_list else np.empty([0, 3])
            np.savetxt(res_fpath, res_list, '%d')
            recall, precision, moda, modp = evaluate(os.path.abspath(res_fpath),
                                                     os.path.abspath(dataloader.dataset.gt_fpath),
                                                     dataloader.dataset.base.__name__)
            print(f'moda: {moda:.1f}%, modp: {modp:.1f}%, prec: {precision:.1f}%, recall: {recall:.1f}%')
        else:
            moda = 0

        print(f'Test, loss: {losses / len(dataloader):.6f}, Time: {t_epoch:.3f}')

        return losses / len(dataloader), moda

    def process_pseudo_gt(self, img_res):
        imgs_heatmap, imgs_offset, imgs_wh, imgs_id = img_res
        imgs_detections = ctdet_decode(imgs_heatmap, imgs_offset, imgs_wh, imgs_id)
        BN, K, _ = imgs_detections.shape
        imgs_detections = imgs_detections.view(BN * K, -1)
        world_xys = self.model.proj_mats * torch.cat([imgs_detections[:, :2],
                                                      torch.ones([BN * K, 1], device=imgs_detections.device)], dim=1)
        world_xys = world_xys[:, :2] / world_xys[:, 2]
