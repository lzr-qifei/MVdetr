from multiview_detector.utils.nested_tensor import NestedTensor, tensor_list_to_nested_tensor
import numpy as np
import random
import math
import cv2
def collate_fn(batch):
    collated_batch = {
        "images": [],
        "infos": [],
        "mats": []
    }
    
    for data in batch:
        # collated_batch["images"].append(data["images"])
        collated_batch["images"].append(data["images"])
        # collated_batch["infos"].append(data["infos"])
        collated_batch["infos"].append(data["infos"])
        collated_batch["mats"].append(data["affine_mats"])
    collated_batch["nested_tensors"] = tensor_list_to_nested_tensor([_ for seq in collated_batch["images"] for _ in seq])
    shape = collated_batch["nested_tensors"].tensors.shape
    # print(shape)
    b = len(batch)
    t = len(collated_batch["images"][0])
    n = shape[1]
    # n = len(collated_batch["images"][1])
    collated_batch["nested_tensors"].tensors = collated_batch["nested_tensors"].tensors.reshape(
        b, t,n, shape[2], shape[3], shape[4]
    )
    collated_batch["nested_tensors"].mask = collated_batch["nested_tensors"].mask.reshape(
        b, t,n, shape[3], shape[4]
    )
    # collated_batch[]
    return collated_batch
