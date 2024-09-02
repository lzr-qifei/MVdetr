
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader
from .SeqDataset import build_mv_dataset
# from ..utils.utils import is_distributed
# from .mot_dataset import build as build_mot_dataset
from .utils import collate_fn
import torch
def is_distributed():
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return False
    return True

def build_dataset(config: dict,train: bool):
    return build_mv_dataset(config,train)


def build_sampler(dataset, shuffle: bool):
    if is_distributed():
        sampler = DistributedSampler(dataset=dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle is True else SequentialSampler(dataset)
    return sampler


def build_dataloader(dataset, sampler, batch_size: int, num_workers: int):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
