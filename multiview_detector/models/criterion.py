##modified form Deformble DETR
##add by lizirui
import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributed as dist
import copy
from multiview_detector.loss import *
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    # print('prob: ',prob[:10])
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher,matcher_batch, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.matcher_batch = matcher_batch
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.regloss = RegL1Loss()
        # self.mseloss = nn.MSELoss()
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print('outlabel_shape: ',src_logits.shape)
        # targets = list(targets)
        if type(targets) != list:
            targets = [targets]

        idx = self._get_src_permutation_idx(indices)
        # print("idx:",idx)
        target_classes_o = torch.cat([t["labels"][0][J] for t, (_, J) in zip(targets, indices)])
        # print(target_classes_o)
        target_classes_o = target_classes_o.long().to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:1], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # print('tgt_cls : ', target_classes)
        target_classes[idx[1]] = target_classes_o


        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]+1],
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
        # print("target_classes_onehot:",target_classes_onehot)
        # target_classes_onehot = target_classes_onehot[:,:-1]
        target_classes_onehot = target_classes_onehot[:,1:]
        # print("src_logits:",src_logits)
        # print("target_classes_onehot:",target_classes_onehot)
        # print("targets:",targets['labels'])
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce': loss_ce}
        losses = {'labels': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, indices, num_boxes):
    #     """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs['pred_logits']
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
    #     # Count the number of predictions that are NOT "no-object" (which is the last class)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'cardinality_error': card_err}
    #     return losses
    def compare_pts(self,src,targets):
        src = src.cpu().detach().numpy()
        # src = src*4
        targets = targets.cpu().detach().numpy()
        src_file = '/root/MVdetr/multiview_detector/results/src.txt'
        tgt_file = '/root/MVdetr/multiview_detector/results/tgt.txt'
        with open(src_file,'w') as f :
            for i in range(src.shape[0]):
                tmp = ' '.join(map(str, src[i,:]))+'\n'
                # tmp = str(src[i,:].item())+'\n'
                f.writelines(tmp)
        with open(tgt_file,'w') as f :
            for i in range(src.shape[0]):
                # tmp = str(targets[i,:])+'\n'
                tmp = ' '.join(map(str, targets[i,:]))+'\n'
                f.writelines(tmp)
        return None
    def loss_center(self,outputs, targets ,indices,log=True):
        # print('indice shape: ',indices)
        if type(targets) != list:
            targets = [targets]
        src_centers = outputs['pred_ct_pts']
        # print('pred_center_shape: ',src_centers.shape)
        idx = self._get_src_permutation_idx(indices)
        src_centers_sorted = src_centers[idx[1]]
        target_centers_o = torch.cat([t["pts"][0][J] for t, (_, J) in zip(targets, indices)])
        # print('tgt_pts_shape: ',target_centers_o.shape)
        # target_centers = torch.full(src_centers.shape[:2], self.num_classes,
        #                     dtype=torch.int64, device=src_centers.device)
        target_centers_o = target_centers_o.to(src_centers.device)
        # target_centers[idx[1]] = target_centers_o
        # loss_center = self.l1loss(src_centers,targets[0]['reg_mask'], targets[0]['idx'], targets[0]['world_pts'])
        # print('mask: ',targets[0]['reg_mask'])
        loss_center = self.regloss(src_centers_sorted,None, None, target_centers_o)
        # print('target pt: ',target_centers_o)
        # loss_center = self.mseloss(src_centers,)
        self.compare_pts(src_centers_sorted,target_centers_o)
        # losses = {'loss_center': loss_center}
        losses = {'center': loss_center}
        return losses
    def loss_offset(self,outputs, targets ,indices,log=True):
        #TODO:存在bug，需要按照center的计算方式修正一下
        targets = [targets]
        src_offsets = outputs['pred_offsets']
        # print('pred_offset_shape: ',src_offsets.shape)
        idx = self._get_src_permutation_idx(indices)
        target_offsets_o = torch.cat([t["offset"][0][J] for t, (_, J) in zip(targets, indices)])
        # print('tgt_ofst_shape: ',target_offsets_o.shape)
        target_offsets = torch.full(src_offsets.shape[:2], self.num_classes,
                            dtype=torch.int64, device=src_offsets.device)
        # target_offsets_o = target_offsets_o.to(src_offsets.device)
        # target_offsets[idx[1]] = target_offsets_o
        loss_offset = self.l1loss(src_offsets,targets[0]['reg_mask'], targets[0]['idx'], target_offsets_o)
        losses = {'loss_offset': loss_offset}
        return losses
    def loss_labels_batch(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_center_batch(self,outputs, targets ,indices,log=True):
        src_centers = outputs['pred_ct_pts']
        # print('pred_center_shape: ',src_centers.shape)
        idx = self._get_src_permutation_idx(indices)
        src_centers_sorted = src_centers[idx]
        target_centers_o = torch.cat([t["pts"][J] for t, (_, J) in zip(targets, indices)])
        # print('tgt_pts_shape: ',target_centers_o.shape)
        # target_centers = torch.full(src_centers.shape[:2], self.num_classes,
        #                     dtype=torch.int64, device=src_centers.device)
        target_centers_o = target_centers_o.to(src_centers.device)
        # target_centers[idx[1]] = target_centers_o
        # loss_center = self.l1loss(src_centers,targets[0]['reg_mask'], targets[0]['idx'], targets[0]['world_pts'])
        # print('mask: ',targets[0]['reg_mask'])
        # loss_center = self.regloss(src_centers_sorted,None, None, target_centers_o)
        loss_center = F.l1_loss(src_centers_sorted,target_centers_o)
        # loss_center = self.mseloss(src_centers,)
        # self.compare_pts(src_centers_sorted,target_centers_o)
        # losses = {'loss_center': loss_center}
        losses = {'center': loss_center}
        return losses
        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'cardinality': self.loss_cardinality,
        #     'boxes': self.loss_boxes,
        #     'masks': self.loss_masks
        # }
        loss_map = {
            'labels':self.loss_labels,
            'center':self.loss_center
            # 'offset':self.loss_offset
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    def get_loss_batch(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'cardinality': self.loss_cardinality,
        #     'boxes': self.loss_boxes,
        #     'masks': self.loss_masks
        # }
        loss_map = {
            'labels':self.loss_labels_batch,
            'center':self.loss_center_batch
            # 'offset':self.loss_offset
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if type(targets) != list:
            indices = self.matcher(outputs_without_aux, targets)
            num_points = len(targets["labels"][0])
            num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_points)
            num_points = torch.clamp(num_points / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))
        else:
            indices = self.matcher_batch(outputs_without_aux, targets)
            num_points = len(targets[0]["labels"]) * len(targets)
            num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_points)
            num_points = torch.clamp(num_points / get_world_size(), min=1).item()

            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                kwargs = {}
                losses.update(self.get_loss_batch(loss, outputs, targets, indices, num_points, **kwargs))
        # print('indices shape: ',indices.shape)
        # print('indices: ',indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_points = sum(len(t["labels"]) for t in targets)
        # print(targets[0]["labels"])
        


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_points, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses,indices
import einops
class IDCriterion(nn.Module):
    def __init__(self, weight: float, gpu_average: bool):
        super().__init__()
        self.weight = weight
        self.gpu_average = gpu_average if gpu_average is not None else None
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, outputs, targets):
        assert len(outputs) == 1, f"ID Criterion is only supported bs=1, but get bs={len(outputs)}"
        outputs = einops.rearrange(outputs, "b n c -> (b n) c")
        targets = einops.rearrange(targets, "b n c -> (b n) c")
        ce_loss = self.ce_loss(outputs, targets).sum()
        # Average:
        num_ids = len(outputs)
        num_ids = torch.as_tensor([num_ids], dtype=torch.float, device=outputs.device)
        # if self.gpu_average:
        #     if is_distributed():
        #         torch.distributed.all_reduce(num_ids)
        #     num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()
        return ce_loss / num_ids
class DETRCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.regloss = RegL1Loss()
        # self.mseloss = nn.MSELoss()
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print('outlabel_shape: ',src_logits.shape)
        # targets = list(targets)
        targets = [targets]

        idx = self._get_src_permutation_idx(indices)
        # print("idx:",idx)
        target_classes_o = torch.cat([t["labels"][0][J] for t, (_, J) in zip(targets, indices)])
        # print(target_classes_o)
        target_classes_o = target_classes_o.long().to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:1], 0,
                                    dtype=torch.int64, device=src_logits.device)
        # print('tgt_cls : ', target_classes)
        target_classes[idx[1]] = target_classes_o


        # target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
        #                                     dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]+1],
                                    dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(1, target_classes.unsqueeze(-1), 1)
        # print("target_classes_onehot:",target_classes_onehot)
        # target_classes_onehot = target_classes_onehot[:,:-1]
        target_classes_onehot = target_classes_onehot[:,1:]
        # print("src_logits:",src_logits)
        # print("target_classes_onehot:",target_classes_onehot)
        # print("targets:",targets['labels'])
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        # losses = {'loss_ce': loss_ce}
        losses = {'labels': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    # @torch.no_grad()
    # def loss_cardinality(self, outputs, targets, indices, num_boxes):
    #     """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
    #     This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
    #     """
    #     pred_logits = outputs['pred_logits']
    #     device = pred_logits.device
    #     tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
    #     # Count the number of predictions that are NOT "no-object" (which is the last class)
    #     card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
    #     card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
    #     losses = {'cardinality_error': card_err}
    #     return losses
    def compare_pts(self,src,targets):
        src = src.cpu().detach().numpy()
        # src = src*4
        targets = targets.cpu().detach().numpy()
        src_file = '/root/MVdetr/multiview_detector/results/src.txt'
        tgt_file = '/root/MVdetr/multiview_detector/results/tgt.txt'
        with open(src_file,'w') as f :
            for i in range(src.shape[0]):
                tmp = ' '.join(map(str, src[i,:]))+'\n'
                # tmp = str(src[i,:].item())+'\n'
                f.writelines(tmp)
        with open(tgt_file,'w') as f :
            for i in range(src.shape[0]):
                # tmp = str(targets[i,:])+'\n'
                tmp = ' '.join(map(str, targets[i,:]))+'\n'
                f.writelines(tmp)
        return None
    def loss_center(self,outputs, targets ,indices,log=True):
        # print('indice shape: ',indices)
        targets = [targets]
        src_centers = outputs['pred_ct_pts']
        # print('pred_center_shape: ',src_centers.shape)
        idx = self._get_src_permutation_idx(indices)
        src_centers_sorted = src_centers[idx[1]]
        target_centers_o = torch.cat([t["world_pts"][0][J] for t, (_, J) in zip(targets, indices)])
        # print('tgt_pts_shape: ',target_centers_o.shape)
        # target_centers = torch.full(src_centers.shape[:2], self.num_classes,
        #                     dtype=torch.int64, device=src_centers.device)
        target_centers_o = target_centers_o.to(src_centers.device)
        # target_centers[idx[1]] = target_centers_o
        # loss_center = self.l1loss(src_centers,targets[0]['reg_mask'], targets[0]['idx'], targets[0]['world_pts'])
        # print('mask: ',targets[0]['reg_mask'])
        loss_center = self.regloss(src_centers_sorted,targets[0]['reg_mask'], targets[0]['idx'], target_centers_o)
        # loss_center = self.mseloss(src_centers,)
        # self.compare_pts(src_centers_sorted,target_centers_o)
        # losses = {'loss_center': loss_center}
        losses = {'center': loss_center}
        return losses
    def loss_offset(self,outputs, targets ,indices,log=True):
        #TODO:存在bug，需要按照center的计算方式修正一下
        targets = [targets]
        src_offsets = outputs['pred_offsets']
        # print('pred_offset_shape: ',src_offsets.shape)
        idx = self._get_src_permutation_idx(indices)
        target_offsets_o = torch.cat([t["offset"][0][J] for t, (_, J) in zip(targets, indices)])
        # print('tgt_ofst_shape: ',target_offsets_o.shape)
        target_offsets = torch.full(src_offsets.shape[:2], self.num_classes,
                            dtype=torch.int64, device=src_offsets.device)
        # target_offsets_o = target_offsets_o.to(src_offsets.device)
        # target_offsets[idx[1]] = target_offsets_o
        loss_offset = self.l1loss(src_offsets,targets[0]['reg_mask'], targets[0]['idx'], target_offsets_o)
        losses = {'loss_offset': loss_offset}
        return losses

        

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # loss_map = {
        #     'labels': self.loss_labels,
        #     'cardinality': self.loss_cardinality,
        #     'boxes': self.loss_boxes,
        #     'masks': self.loss_masks
        # }
        loss_map = {
            'labels':self.loss_labels,
            'center':self.loss_center,
            'offset':self.loss_offset
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        # print('indices shape: ',indices.shape)
        # print('indices: ',indices)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # num_points = sum(len(t["labels"]) for t in targets)
        num_points = len(targets["labels"][0])
        num_points = torch.as_tensor([num_points], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_points)
        num_points = torch.clamp(num_points / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_points, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_points, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_points, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses,indices