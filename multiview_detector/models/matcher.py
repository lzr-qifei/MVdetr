# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from multiview_detector.utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 0.1,
                 cost_pts: float = 2,
                 cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_pts = cost_pts
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_pts != 0 or cost_giou != 0, "all costs cant be 0"

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            targets = [targets]
            # bs, num_queries = outputs["pred_logits"].shape[:2]
            num_queries,_ = outputs["pred_logits"].shape[:2]
            # We flatten to compute the cost matrices in a batch
            # out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_prob = outputs["pred_logits"].sigmoid()
            # print('out_prob shape: ',out_prob.shape)
            out_pts = outputs['pred_ct_pts'].cpu()
            # out_bbox = outputs["pred_boxes"].flatten(0, 1)  
            # out_pts = outputs['pred_ct_pts'].flatten(0,1)# [batch_size * num_queries, 2]
            # test_pts = outputs['pred_ct_pts']
            # print('out_pt shape: ',test_pts.shape)
            # test_cls = out_prob

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            # print('tgtid: ',tgt_ids.shape)
            tgt_ids = tgt_ids.long()
            # tgt_bbox = torch.cat([v["boxes"] for v in targets])
            tgt_pts = torch.cat([v["world_pts"] for v in targets])
            tgt_pts = tgt_pts.float()
            # print('tgt_pt:',tgt_pts.shape)

            # Compute the classification cost.
            alpha = 0.25
            gamma = 2.0
            neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            # print('pos_c_cls: ',pos_cost_class.shape)
            # print('tgt_ids: ',tgt_ids[0])
            # cost_class = pos_cost_class[:, tgt_ids[0]] - neg_cost_class[:, tgt_ids[0]]
            cost_class = pos_cost_class - neg_cost_class
            # cost_class = pos_cost_class[:,0] - neg_cost_class[:, 0]
            # cost_class = pos_cost_class[tgt_ids[0], :] - neg_cost_class[tgt_ids[0], :]
            cost_class = cost_class.cpu()
            # print('cost_class shape: ',cost_class.shape)
            # print('cost_class: ',cost_class)
            # cost_class = 0

            # Compute the L1 cost between boxes
            # cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            cost_pts = torch.cdist(out_pts,tgt_pts[0],p=2)
            # print('cost_pts shape: ',cost_pts.shape)
            # print('cost_pts: ',cost_pts)
            # Compute the giou cost betwen boxes
            # cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                            #  box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            # C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = self.cost_pts * cost_pts + self.cost_class * cost_class 
            # C = C.view(bs, num_queries, -1).cpu()
            C = C.view( num_queries, -1)
            # print('cost',C.shape)

            # sizes = [len(v["world_pts"]) for v in targets]
            # sizes = len(tgt_pts[0])
            # print('sizes: ',sizes)
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C)]
            indices = [linear_sum_assignment(C)]
            # print('indices: ',indices)
            print('indices0 shape: ',len(indices[0][0]))
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_pts=args.set_cost_bbox,
                            cost_giou=args.set_cost_giou)
