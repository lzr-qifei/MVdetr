# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from multiview_detector.utils.tensor_utils import _transpose_and_gather_feat, _sigmoid
import torch.nn.functional as F


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, output, target, mask=None):
        ''' Modified focal loss. Exactly the same as CornerNet.
            Runs faster and costs a little bit more memory
          Arguments:
            output (batch x c x h x w)
            target (batch x c x h x w)
        '''
        if mask is None:
            mask = torch.ones_like(target)
        output = _sigmoid(output)
        target = target.to(output.device)
        mask = mask.to(output.device)
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        print('pos inds: ',pos_inds)
        print('neg inds: ',neg_inds)

        neg_weights = torch.pow(1 - target, 4)

        pos_loss = torch.log(output) * torch.pow(1 - output, 2) * pos_inds
        neg_loss = torch.log(1 - output) * torch.pow(output, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = (neg_loss * mask).sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # print(output.device,mask.device,ind.device,target.device)
        if target.device=='cpu':
            # mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
            target = target.to(output.device)
        else:
            # mask, ind= mask.to(output.device), ind.to(output.device)
            pass
        # print(output.device,mask.device,ind.device,target.device)
        # print('mask: ',mask)
        # output = output.unsqueeze(0)
        # output = output.unsqueeze(0)
        # pred = _transpose_and_gather_feat(output, ind)
        pred = output
        # pred[:,0]*=250
        # pred[:,1]*=160
        # target[:,0]*=250
        # target[:,1]*=160
        # mask = mask.unsqueeze(2).expand_as(pred).float()
        # print('pred:',pred.shape)
        # print('target: ',target.shape)
        # loss = F.l1_loss(pred , target , reduction='sum')
        # if type =='mse':
        #     loss = F.mse_loss(pred,target,reduction='mean')
        # else:
        loss = F.l1_loss(pred , target , reduction='mean')
        # loss = loss / (mask.sum() + 1e-4)
        return loss


class RegCELoss(nn.Module):
    def __init__(self):
        super(RegCELoss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask, ind, target = mask.to(output.device), ind.to(output.device), target.to(output.device)
        pred = _transpose_and_gather_feat(output, ind)
        if len(target[mask]) != 0:
            loss = F.cross_entropy(pred[mask], target[mask], reduction='sum')
            loss = loss / (mask.sum() + 1e-4)
        else:
            loss = 0
        return loss
