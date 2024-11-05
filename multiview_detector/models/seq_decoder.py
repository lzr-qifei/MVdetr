# Copyright (c) RuopengGao. All Rights Reserved.
# About:
import os
import copy
import einops
import torch
import torch.nn as nn
import random
import math

from structures.instances import Instances
from utils.utils import labels_to_one_hot, pos_to_pos_embed
# from .trajectory_modeling import TrajectoryAugmentation
from multiview_detector.models.id_decoder import IDDecoder
# from .ffn import FFN
from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from typing import List


from utils.utils import get_activation_layer


class FFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout: float, activation: str = "ReLU"):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        # self.activation = nn.ReLU(inplace=True)
        self.activation = get_activation_layer(activation=activation)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, gate=None):
        tgt2 = self.linear2(
            self.dropout1(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )
        if gate is None:    # without gated control:
            tgt = tgt + self.dropout2(tgt2)
        else:
            tgt = tgt + gate.tanh() * self.dropout2(tgt2)
        tgt = self.norm(tgt)
        return tgt
# import configs.runtime as runtime


class SeqDecoder(nn.Module):
    def __init__(
            self,
            detr_hidden_dim: int,
            hidden_dim: int,                # the dim for each field
            dim_feedforward: int,           # the ffn_dim for each field
            num_heads: int,                 # the n_heads for each field
            dropout: float,
            n_id_decoder_layers: int,
            num_id_vocabulary: int,
            training_num_id: int,
            device: str,
            max_temporal_length: int,
            multi_times_id_decoder: int,
    ):
        super().__init__()
        self.detr_hidden_dim = detr_hidden_dim
        self.hidden_dim = hidden_dim
        self.dim_feedforward = dim_feedforward
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_id_decoder_layers = n_id_decoder_layers
        self.num_id_vocabulary = num_id_vocabulary
        self.training_num_id = training_num_id
        self.device = device
        self.max_temporal_length = max_temporal_length
        self.multi_times_id_decoder = multi_times_id_decoder
        assert self.multi_times_id_decoder == 0 or self.multi_times_id_decoder >= 2, \
            f"Currently, we only support multi_times_id_decoder == 0 OR >= 2, but get {self.multi_times_id_decoder}."

        assert self.detr_hidden_dim == self.hidden_dim, f"Currently, we only support the same hidden dim."

        self.trajectory_feature_adapter = FFN(
            d_model=self.hidden_dim,
            d_ffn=self.dim_feedforward,
            dropout=self.dropout,
            activation="GELU"
        )

        # self.trajectory_augmentation = TrajectoryAugmentation(
        #     hidden_dim=self.hidden_dim,
        #     dim_feedforward=self.dim_feedforward,
        #     dropout=self.dropout,
        #     device=self.device,
        # )

        self.id_decoder = IDDecoder(
            hidden_dim=self.hidden_dim,
            dim_feedforward=self.dim_feedforward,
            n_heads=self.num_heads,
            dropout=self.dropout,
            n_layers=self.n_id_decoder_layers,
            device=self.device,
            num_id_vocabulary=self.num_id_vocabulary,
            max_temporal_length=self.max_temporal_length,
        ).to(self.device)

        return

    def forward(
            self,
            track_seqs: List[List[Instances]],
            traj_drop_ratio: float = 0.0,
            traj_switch_ratio: float = 0.0,
            use_checkpoint: bool = False,
            inference_ensemble: int = 0,
    ):
        if self.multi_times_id_decoder == 0 or not self.training:
            # Original training and inference:
            if self.training or inference_ensemble == 0:
                format_seqs = self.prepare(
                    track_seqs=track_seqs,
                    traj_drop_ratio=traj_drop_ratio,
                    traj_switch_ratio=traj_switch_ratio,
                )
                pred_id_words, gt_id_words = self.net_forward(
                    format_seqs=format_seqs,
                    use_checkpoint=use_checkpoint,
                )
            else:
                assert inference_ensemble >= 2
                format_seqs = self.prepare(
                    track_seqs=track_seqs,
                    traj_drop_ratio=traj_drop_ratio,
                    traj_switch_ratio=traj_switch_ratio,
                )
                pred_id_words = []
                gt_id_words = None
                for ensemble_i in range(inference_ensemble):
                    g = torch.Generator()
                    g.manual_seed(ensemble_i)
                    _id_shuffle = torch.randperm(self.training_num_id, generator=g).to(self.device)
                    _format_seqs = copy.deepcopy(format_seqs)
                    _N, _T = _format_seqs[0]["trajectory"]["ids"].shape
                    _shuffled_ids = _id_shuffle[
                        einops.rearrange(_format_seqs[0]["trajectory"]["ids"], "n t -> (n t)")
                    ]
                    _shuffled_ids = einops.rearrange(_shuffled_ids, "(n t) -> n t", n=_N, t=_T)
                    _format_seqs[0]["trajectory"]["ids"] = _shuffled_ids
                    _pred_id_words, _ = self.net_forward(
                        format_seqs=_format_seqs,
                        use_checkpoint=False,
                    )
                    _pred_id_words = _pred_id_words[:, :, torch.cat(
                        [_id_shuffle, torch.tensor([self.num_id_vocabulary], device=self.device, dtype=torch.long)]
                    )]
                    pred_id_words.append(_pred_id_words)
        else:
            # print('******************')
            pred_id_words, gt_id_words = None, None
            for _time in range(self.multi_times_id_decoder):
                assert len(track_seqs) == 1, f"SeqDecoder currently only support BS=1, but get BS={len(track_seqs)}."
                _track_seqs = [[
                    _.clone() for _ in track_seqs[0]
                ]]
                # Random the IDs in the track_seqs:
                _id_map = torch.randperm(self.training_num_id).to(self.device)
                for t in range(len(_track_seqs[0])):
                    _track_seqs[0][t].ids = _id_map[_track_seqs[0][t].ids]
                format_seqs = self.prepare(
                    track_seqs=_track_seqs,
                    traj_drop_ratio=traj_drop_ratio,
                    traj_switch_ratio=traj_switch_ratio,
                )
                _pred_id_words, _gt_id_words = self.net_forward(
                    format_seqs=format_seqs,
                    use_checkpoint=use_checkpoint,
                )
                if pred_id_words is None:
                    pred_id_words = _pred_id_words
                    gt_id_words = _gt_id_words
                else:
                    pred_id_words = torch.cat([pred_id_words, _pred_id_words], dim=1)
                    gt_id_words = torch.cat([gt_id_words, _gt_id_words], dim=1)

        return pred_id_words, gt_id_words

        return pred_id_words, gt_id_words

    def prepare(
            self,
            # The whole track_seqs, which contains the current detection results:
            track_seqs: List[List[Instances]],
            # The training augmentation parameters:
            traj_drop_ratio: float = 0.0,
            traj_switch_ratio: float = 0.0,
    ):
        assert len(track_seqs) == 1, f"SeqDecoder currently only support BS=1, but get BS={len(track_seqs)}."
        track_seq = track_seqs[0]       # for simplicity, we only use the first one

        # Here are some symbols we use:
        # N is the number of targets in the track_seq;
        # T is the temporal length of the track_seq;

        # All information is stored in a corresponding T-len list.
        all_ids = [_.id_words for _ in track_seq] if self.training else [_.ids for _ in track_seq]
        all_features = [_.outputs for _ in track_seq]
        # all_boxes = [_.gt_boxes.detach().to(all_features[0].device) for _ in track_seq] if self.training \
        #     else [_.boxes.detach() for _ in track_seq]
        all_pts = [_.gt_pts.detach().to(all_features[0].device) for _ in track_seq] if self.training \
            else [_.pts.detach() for _ in track_seq]
        T = len(all_ids)
        # print('all id: ',T)
        feature_dim = all_features[0].shape[-1]
        pt_dim = all_pts[0].shape[-1]
        device = all_features[0].device

        # Statistics of IDs that appear in the track_seq:
        all_ids_in_one_list = torch.cat(all_ids, dim=0).tolist()
        all_ids_set = set(all_ids_in_one_list)
        all_ids_set.discard(self.num_id_vocabulary)     # exclude the special ID token
        N = len(all_ids_set)

        # Build a mapping from ID to index, and index to ID:
        id_to_idx = {list(all_ids_set)[_]: list(range(N))[_] for _ in range(N)}
        idx_to_id = {v: k for k, v in id_to_idx.items()}

        # Prepare the historical trajectory fields,
        # which should be in (N, T-1) shape.
        trajectory_ids_list, trajectory_features_list, trajectory_pts_list, trajectory_times_list = [], [], [], []
        trajectory_masks_list = []
        idxs_temp = {}
        # Generate the historical trajectory fields:
        for t in range(T - 1):  # the historical trajectory only contains T-1 frames
            t_idxs = torch.tensor(
                [id_to_idx[_id.item()] for _id in all_ids[t]], dtype=torch.long, device=device
            )   # which index to use, for each object in current frame "t"
            idxs_temp[t] = t_idxs
            t_token_mask = torch.ones((N, ), dtype=torch.bool, device=device)
            t_token_mask[t_idxs] = False    # in our code, False means the token is valid, True means invalid
            t_times = t * torch.ones((N, ), dtype=torch.long, device=device)
            # Init fields:
            t_ids = -torch.ones((N, ), dtype=torch.long, device=device)
            t_features = torch.zeros((N, feature_dim), dtype=torch.float, device=device)
            t_pts = torch.zeros((N, pt_dim), dtype=torch.float, device=device)
            # Fill fields:
            t_ids[t_idxs] = all_ids[t].to(device)
            # print('t_ids shape: ',t_ids.shape)
            t_features[t_idxs] = all_features[t]
            t_pts[t_idxs] = all_pts[t]
            # Append to the list:
            trajectory_ids_list.append(t_ids)
            trajectory_features_list.append(t_features)
            trajectory_pts_list.append(t_pts)
            trajectory_times_list.append(t_times)
            trajectory_masks_list.append(t_token_mask)
        # Stack the historical trajectory fields into tensors,
        # shape=(N, T-1, ...)
        trajectory_features = torch.stack(trajectory_features_list, dim=1)
        trajectory_pts = torch.stack(trajectory_pts_list, dim=1)
        trajectory_times = torch.stack(trajectory_times_list, dim=1)
        trajectory_ids = torch.stack(trajectory_ids_list, dim=1)
        trajectory_masks = torch.stack(trajectory_masks_list, dim=1)

        # print('trajectory_features: ',trajectory_features.shape)
        # print('trajectory_boxes: ',trajectory_boxes.shape)
        # print('trajectory_ids: ',trajectory_ids.shape)
        # print('trajectory_ids: ',trajectory_ids)
        # print('trajectory_times: ',trajectory_times)
        # Prepare the current detection fields,
        # they have nearly the same attributes as historical trajectories.
        # We denote they as "unknown" because they need to be decoded.
        unknown_features_list, unknown_pts_list, unknown_ids_list, unknown_times_list = [], [], [], []
        unknown_masks_list = []
        unknown_id_gts_list: list | None = [] if self.training else None

        if self.training:
            # During training, the last T-1 frames will be used to supervise the model,
            # so they are all "unknown".
            for t in range(1, T):
                N_t = len(all_ids[t])   # how many objects in this frame
                # Init fields:
                t_token_mask = torch.ones((N,), dtype=torch.bool, device=device)
                t_ids = -torch.ones((N,), dtype=torch.long, device=device)
                t_features = torch.zeros((N, feature_dim), dtype=torch.float, device=device)
                t_pts = torch.zeros((N, pt_dim), dtype=torch.float, device=device)
                t_times = t * torch.ones((N,), dtype=torch.long, device=device)
                t_id_gts = -torch.ones((N,), dtype=torch.long, device=device)
                # Fill fields:
                if t in idxs_temp:
                    t_idxs = idxs_temp[t]   # this would be faster, but insignificant
                else:
                    t_idxs = torch.tensor(
                        [id_to_idx[_id.item()] for _id in all_ids[t]], dtype=torch.long, device=device
                    )
                t_token_mask[t_idxs] = False
                t_ids[t_idxs] = torch.tensor([self.num_id_vocabulary] * N_t, dtype=torch.long, device=device)
                t_features[t_idxs] = all_features[t]
                t_pts[t_idxs] = all_pts[t]
                t_id_gts[t_idxs] = track_seq[t].id_labels.to(device)
                # Append to the list:
                unknown_id_gts_list.append(t_id_gts)
                unknown_masks_list.append(t_token_mask)
                unknown_times_list.append(t_times)
                unknown_ids_list.append(t_ids)
                unknown_pts_list.append(t_pts)
                unknown_features_list.append(t_features)
                # print('unknown_id_gts_list: ',unknown_id_gts_list.shape)
        else:
            # During inference, only the last frame will be used to decode.
            # And the number of objects in the last frame may be different from the previous frames,
            # so we need to redefine N_.
            N_ = len(all_features[-1])
            unknown_features_list.append(all_features[-1])
            unknown_ids_list.append(all_ids[-1].to(device))
            unknown_pts_list.append(all_pts[-1])
            unknown_times_list.append(torch.tensor([T-1] * N_, dtype=torch.long, device=device))
            unknown_masks_list.append(torch.zeros((N_, ), dtype=torch.bool, device=device))
        # Stack the current detection fields into tensors,
        unknown_features = torch.stack(unknown_features_list, dim=1)
        unknown_pts = torch.stack(unknown_pts_list, dim=1)
        unknown_ids = torch.stack(unknown_ids_list, dim=1)
        unknown_times = torch.stack(unknown_times_list, dim=1)
        unknown_masks = torch.stack(unknown_masks_list, dim=1)
        # print('unknown_ids: ',unknown_ids.shape)
        # print('unknown_ids: ',unknown_ids)
        unknown_id_gts = None if unknown_id_gts_list is None else torch.stack(unknown_id_gts_list, dim=1)

        # Training Augmentation:
        # if self.training:
        #     N, T = trajectory_features.shape[0], trajectory_features.shape[1] + 1
        #     # Record which token is removed during this process:
        #     trajectory_remove_masks = torch.zeros((N, T - 1), dtype=torch.bool, device=device)

        #     # Trajectory Token Drop:
        #     for n in range(N):
        #         if random.random() < traj_drop_ratio:
        #             traj_begin = random.randint(0, T-1)
        #             traj_max_t = T - 1 - traj_begin
        #             traj_end = traj_begin + math.ceil(traj_max_t * random.random())
        #             trajectory_remove_masks[n, traj_begin: traj_end] = True
        #     unknown_remove_masks = torch.cat([
        #         trajectory_remove_masks[:, 1:],
        #         torch.zeros((N, 1), dtype=torch.bool, device=device)
        #     ], dim=1)
        #     # Check if the trajectory augmentation process is legal.
        #     # Specifically, we need to ensure there is at least one object can be supervised,
        #     # or it may cause grad == None.
        #     # TODO: This legal check is just a simple implementation, it may not be rigorous.
        #     #       But it's enough for the current code.
        #     is_legal = (~(trajectory_masks | trajectory_remove_masks) & ~(unknown_masks | unknown_remove_masks)).any().item()
        #     if is_legal:
        #         # We do not need to truly remove these tokens,
        #         # just set their to invalid tokens by masks.
        #         trajectory_masks = trajectory_masks | trajectory_remove_masks
        #         unknown_masks = unknown_masks | unknown_remove_masks
        #         # Also, we need to modify some ID ground-truths of "unknown" objects.
        #         # For example, a token first appears at t=3,
        #         # if I remove it at t=3, then I need to modify its ID GT at t=4 to the special token (newborn).
        #         for n in range(N):
        #             line_traj_mask = trajectory_masks[n]
        #             if line_traj_mask[0].item():
        #                 new_born_t = 0
        #                 for _ in line_traj_mask:
        #                     if _.item():
        #                         new_born_t += 1
        #                     else:
        #                         break
        #                 unknown_id_gts[n][:new_born_t] = self.num_id_vocabulary

        #     # Trajectory Token Switch (Swap):
        #     if traj_switch_ratio > 0.0:
        #         for t in range(0, T-1):
        #             switch_p = torch.ones((N, ), dtype=torch.float, device=device) * traj_switch_ratio
        #             switch_map = torch.bernoulli(switch_p)
        #             switch_idxs = torch.nonzero(switch_map)     # objects to switch
        #             switch_idxs = switch_idxs.reshape((switch_idxs.shape[0], ))
        #             if len(switch_idxs) == 1 and N > 1:
        #                 # Only one object can be switched, but we have more than one object.
        #                 # So we need to randomly select another object to switch.
        #                 switch_idxs = torch.as_tensor([switch_idxs[0].item(), random.randint(0, N - 1)], dtype=torch.long, device=device)
        #             if len(switch_idxs) > 1:
        #                 # Switch the trajectory features, boxes and masks:
        #                 shuffle_switch_idxs = switch_idxs[torch.randperm(len(switch_idxs)).to(device)]
        #                 trajectory_features[switch_idxs, t, :] = trajectory_features[shuffle_switch_idxs, t, :]
        #                 trajectory_pts[switch_idxs, t, :] = trajectory_pts[shuffle_switch_idxs, t, :]
        #                 trajectory_masks[switch_idxs, t] = trajectory_masks[shuffle_switch_idxs, t]
        #             else:
        #                 continue    # no object to switch

        return [{
            "trajectory": {
                "features": trajectory_features,
                "pts": trajectory_pts,
                "ids": trajectory_ids,
                "times": trajectory_times,
                "masks": trajectory_masks,
                "pad_masks": None,
            },
            "unknown": {
                "features": unknown_features,
                "pts": unknown_pts,
                "ids": unknown_ids,
                "times": unknown_times,
                "masks": unknown_masks,
                "id_gts": unknown_id_gts
            }
        }]

    def net_forward(
            self,
            format_seqs,
            use_checkpoint,
    ):
        assert len(format_seqs) == 1, f"Only BS=1 is supported, but get BS={len(format_seqs)}"
        format_seq = format_seqs[0]

        # Adapter
        trajectory_feature_embeds = format_seq["trajectory"]["features"]
        unknown_features = format_seq["unknown"]["features"]

        # Use a simple adapter:
        self.trajectory_feature_adapter.to(self.device)
        trajectory_feature_embeds = self.trajectory_feature_adapter(trajectory_feature_embeds)

        # Trajectory Augmentation, only use FFN for now:
        # trajectory_feature_embeds = self.trajectory_augmentation(
        #     seq_features=trajectory_feature_embeds,
        # )

        # Decoding the ID from the trajectory and current detections:
        id_words = self.id_decoder(
            trajectory_feature_embeds=trajectory_feature_embeds,
            trajectory_masks=format_seq["trajectory"]["masks"],
            trajectory_ids=format_seq["trajectory"]["ids"],
            trajectory_times=format_seq["trajectory"]["times"],
            unknown_features=unknown_features,
            unknown_times=format_seq["unknown"]["times"],
            unknown_ids=format_seq["unknown"]["ids"],
            unknown_masks=format_seq["unknown"]["masks"],
            use_checkpoint=use_checkpoint,
        )

        id_gts = format_seq["unknown"]["id_gts"]
        if id_gts is not None:
            id_mask_flatten = einops.rearrange(format_seq["unknown"]["masks"], "n t -> (n t)")
            legal_id_gts = einops.rearrange(id_gts, "n t -> (n t)")[~id_mask_flatten]
            id_gt_words = labels_to_one_hot(legal_id_gts, class_num=self.num_id_vocabulary+1, device=self.device)
        else:
            id_gt_words = None

        return id_words[None, ...], id_gt_words[None, ...] if id_gt_words is not None else None

    def get_temporal_embeds(self, n_targets_in_frames: list):
        temporals = []
        for i in range(len(n_targets_in_frames)):
            temporals += [i] * n_targets_in_frames[i]
        temporals = torch.tensor(temporals, device=self.device, dtype=torch.float).reshape(-1, 1)
        temporal_embeds = pos_to_pos_embed(pos=temporals, num_pos_feats=256)
        temporal_embeds = self.temporal_embed_head(temporal_embeds)
        return temporal_embeds
