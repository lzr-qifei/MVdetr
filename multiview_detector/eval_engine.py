import os
from utils.utils import yaml_to_dict, is_main_process, distributed_rank, set_seed,infos_to_detr_targets,batch_iterator,combine_detr_outputs,resize_detr_outputs
import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader
from structures.instances import Instances
# @torch.no_grad()
from structures.ordered_set import OrderedSet
from collections import deque
import numpy as np

    
@torch.no_grad()
def submit_one_seq(
            model: nn.Module, dataloader, seq_dir: str, outputs_dir: str,
            only_detr: bool, max_temporal_length: int = 0,
            det_thresh: float = 0.5, newborn_thresh: float = 0.5, id_thresh: float = 0.1,
            image_max_size: int = 1333,
            fake_submit: bool = False,
            inference_ensemble: int = 0,
        ):
    os.makedirs(os.path.join(outputs_dir, "tracker","pred"), exist_ok=True)
    # seq_dataset = SeqDataset(seq_dir=seq_dir, dataset=dataset, width=image_max_size)

    # dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    # seq_name = seq_dir.split("/")[-1]
    seq_name = os.path.split(seq_dir)[-1]
    device = model.device
    current_id = 0
    ids_to_results = {}
    id_deque = OrderedSet()     # an ID deque for inference, the ID will be recycled if the dictionary is not enough.
    detr_outputs = None
    # Trajectory history:
    if only_detr:
        trajectory_history = None
    else:
        trajectory_history = deque(maxlen=max_temporal_length)

    if fake_submit:
        print(f"[Fake] Start >> Submit seq {seq_name.split('/')[-1]}, {len(dataloader)} frames ......")
    else:
        print(f"Start >> Submit seq {seq_name.split('/')[-1]}, {len(dataloader)} frames ......")

    # for i, (image, ori_image) in enumerate(seq_dataloader):
    for idx,batch in enumerate(dataloader):
        mota_pred_list = []
        # ori_h, ori_w = ori_image.shape[1], ori_image.shape[2]
        # frames = batch["nested_tensors"]
        # frames.tensors = rearrange(frames.tensors, "b t n c h w -> (b t) n c h w")
        # frames.mask = rearrange(frames.mask, "b t n h w -> (b t) n h w")
        # T = frames.tensors.shape[0]
        frames = batch["images"]
        # mats = batch["affine_mats"]
        T = len(frames)
        for i in range(T):
            # affinemats = batch["mats"][0][i].unsqueeze(0)
            affinemats = batch["affine_mats"][i].unsqueeze(0)
            cur_infer_frame = frames[i].to(device=device)
            detr_infer_outputs = model(cur_infer_frame,affinemats)
            detr_outputs = resize_detr_outputs(detr_infer_outputs)
            # detr_outputs = detr_infer_outputs
            # detr_outputs = combine_detr_outputs(detr_infer_outputs,detr_outputs)
        
            detr_logits = detr_outputs["pred_logits"]
            detr_scores = torch.max(detr_logits, dim=-1).values.sigmoid()
            detr_det_idxs = detr_scores > det_thresh        # filter by the detection threshold
            detr_det_logits = detr_logits[detr_det_idxs]
            detr_det_labels = torch.max(detr_det_logits, dim=-1).indices
            detr_det_pts = detr_outputs["pred_ct_pts"][detr_det_idxs]
            detr_det_outputs = detr_outputs["outputs"][detr_det_idxs]   # detr output embeddings


            # De-normalize to target image size:
            pts_results = detr_det_pts.cpu() * torch.tensor([1000,640])
            confs_results = detr_det_logits.sigmoid()
            # box_results = box_cxcywh_to_xyxy(boxes=box_results)

            if only_detr is False:
                if len(pts_results) > model.num_id_vocabulary:
                    print(f"[Carefully!] we only support {model.num_id_vocabulary} ids, "
                        f"but get {len(pts_results)} detections in seq {seq_name.split('/')[-1]} {i+1}th frame.")

            # Decoding the current objects' IDs
            if only_detr is False:
                assert max_temporal_length - 1 > 0, f"MOTIP need at least T=1 trajectory history, " \
                                                    f"but get T={max_temporal_length - 1} history in Eval setting."
                current_tracks = Instances(image_size=(0, 0))
                current_tracks.pts = detr_det_pts
                current_tracks.outputs = detr_det_outputs
                current_tracks.ids = torch.tensor([model.num_id_vocabulary] * len(current_tracks),
                                                dtype=torch.long, device=current_tracks.outputs.device)
                current_tracks.confs = detr_det_logits.sigmoid()
                trajectory_history.append(current_tracks)
                if len(trajectory_history) == 1:    # first frame, do not need decoding:
                    newborn_filter = (trajectory_history[0].confs > newborn_thresh).reshape(-1, )   # filter by newborn
                    trajectory_history[0] = trajectory_history[0][newborn_filter]
                    pts_results = pts_results[newborn_filter.cpu()]
                    confs_results = confs_results[newborn_filter.cpu()]
                    ids = torch.tensor([current_id + _ for _ in range(len(trajectory_history[-1]))],
                                    dtype=torch.long, device=current_tracks.outputs.device)
                    trajectory_history[-1].ids = ids
                    for _ in ids:
                        ids_to_results[_.item()] = current_id
                        current_id += 1
                    id_results = []
                    for _ in ids:
                        id_results.append(ids_to_results[_.item()])
                        id_deque.add(_.item())
                    id_results = torch.tensor(id_results, dtype=torch.long)
                else:
                    ids, trajectory_history, ids_to_results, current_id, id_deque, pts_keep = model.inference(
                        trajectory_history=trajectory_history,
                        num_id_vocabulary=model.num_id_vocabulary,
                        ids_to_results=ids_to_results,
                        current_id=current_id,
                        id_deque=id_deque,
                        id_thresh=id_thresh,
                        newborn_thresh=newborn_thresh,
                        inference_ensemble=inference_ensemble,
                    )   # already update the trajectory history/ids_to_results/current_id/id_deque in this function
                    id_results = []
                    for _ in ids:
                        id_results.append(ids_to_results[_])
                    id_results = torch.tensor(id_results, dtype=torch.long)
                    if pts_keep is not None:
                        pts_results = pts_results[pts_keep.cpu()]
                        confs_results = confs_results[pts_keep.cpu()]
            else:   # only detr, ID is just +1 for each detection.
                id_results = torch.tensor([current_id + _ for _ in range(len(pts_results))], dtype=torch.long)
                current_id += len(id_results)

            # Output to tracker file:
            if fake_submit is False:
                # Write the outputs to the tracker file:
                
                # with open(result_file_path, "w") as file:
                assert len(id_results) == len(pts_results)==len(confs_results), f"Pts and IDs should in the same length, " \
                                                            f"but get len(IDs)={len(id_results)} and " \
                                                            f"len(Pts)={len(pts_results)}"
                for obj_id, pt ,conf in zip(id_results, pts_results,confs_results):
                    obj_id = int(obj_id.item())
                    x, y = pt.tolist()
                    frame_id = 359+idx*10+i+1
                    mota_pred_list.extend([[idx,frame_id,obj_id,-1,-1,-1,-1,conf.item(),x,y,-1]])
                        # if dataset in ["DanceTrack", "MOT17", "SportsMOT", "MOT17_SPLIT", "MOT15", "MOT15_V2"]:
                        # if dataset in ["MultiviewX","WildTrack"]:
                        # result_line = f"{idx}," \
                        #             f"{i + 1}," \
                        #             f"{obj_id}," \
                        #             f"-1,-1,-1,-1," \
                        #             f"{conf.item()}," \
                        #             f"{x},{y},-1\n"
                            #seq, frame, id, -1,-1,-1,-1, conf, x, y, -1
                        # else:
                        #     raise NotImplementedError(f"Do not know the outputs format of dataset '{dataset}'.")
                        # file.write(result_line)
        result_file_path = os.path.join(outputs_dir, "tracker","pred", f"mota_pred{idx}.txt")
        np.savetxt(result_file_path,np.array(mota_pred_list), '%f', delimiter=',')
    if fake_submit:
        print(f"[Fake] Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    else:
        print(f"Finish >> Submit seq {seq_name.split('/')[-1]}. ")
    return os.path.join(outputs_dir, "tracker","pred")
def get_seq_names(data_root: str, dataset: str, data_split: str):
    if dataset in ["DanceTrack", "SportsMOT", "MOT17", "MOT17_SPLIT"]:
        dataset_dir = os.path.join(data_root, dataset, data_split)
        return sorted(os.listdir(dataset_dir))
    else:
        raise NotImplementedError(f"Do not support dataset '{dataset}' for eval dataset.")
