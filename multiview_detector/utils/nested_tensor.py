
import torch

from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Args:
            tensors: Tensor, (B, C, H, W)
            mask: Tensor, (B, H, W)
        """
        assert tensors.shape[0] == mask.shape[0], \
            f"tensors have batch size {tensors.shape[0]} but get {mask.shape[0]} for mask."
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        """
        Args:
            device: Device
            non_blocking:
        """
        tensors = self.tensors.to(device, non_blocking=non_blocking)
        if self.mask is None:
            masks = None
        else:
            masks = self.mask.to(device, non_blocking=non_blocking)
        return NestedTensor(tensors=tensors, mask=masks)

    def decompose(self) :
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, item):
        return NestedTensor(tensors=self.tensors[item], mask=self.mask[item])


def tensor_list_to_nested_tensor(tensor_list: List[torch.Tensor], size_divisibility: int = 32) -> NestedTensor:
    """
    Args:
        tensor_list:
        size_divisibility:

    Returns:
    """
    for t in tensor_list:
        t = t.squeeze() 
    assert tensor_list[0].dim() == 4, f"Tensor should have 4 dimensions, but get {tensor_list[0].dim()}"
    heights, widths = zip(*[t.shape[2:] for t in tensor_list])
    final_shape = [len(tensor_list)] + [tensor_list[0].shape[0]]+ [tensor_list[0].shape[1]] + list(map(max, (heights, widths)))
    final_b, final_n,final_c, final_h, final_w = final_shape
    # print(final_b, final_n,final_c, final_h, final_w)
    # if size_divisibility > 0:
    #     stride = size_divisibility
    #     final_h = (final_h + (stride - 1)) // stride * stride
    #     final_w = (final_w + (stride - 1)) // stride * stride
    final_shape = [final_b, final_n,final_c, final_h, final_w]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensors = torch.zeros(final_shape, dtype=dtype, device=device)
    masks = torch.ones((final_b, final_n,final_h, final_w), dtype=torch.bool, device=device)
    for input_tensor, pad_tensor, mask in zip(tensor_list, tensors, masks):
        assert input_tensor.shape[1] == final_shape[2], "Tensor channel size should be equal."
        pad_tensor[: input_tensor.shape[0], : input_tensor.shape[1], : input_tensor.shape[2], : input_tensor.shape[3]].copy_(input_tensor)
        mask[: input_tensor.shape[2], : input_tensor.shape[3]] = False
    return NestedTensor(tensors=tensors, mask=masks)


def nested_tensor_index_select(nested_tensor: NestedTensor, dim: int, index: torch.Tensor):
    tensors, mask = nested_tensor.decompose()
    selected_tensors = torch.index_select(input=tensors, dim=dim, index=index).contiguous()
    selected_mask = torch.index_select(input=mask, dim=dim, index=index).contiguous()
    return NestedTensor(tensors=selected_tensors, mask=selected_mask)

