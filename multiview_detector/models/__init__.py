import torch

from .tracker import build 
# from utils.utils import distributed_rank


def build_model(config: dict):
    model = build(config=config)
    model.to(device=torch.device(config["DEVICE"]))
    return model