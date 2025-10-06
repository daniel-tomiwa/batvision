import torch
import numpy as np
import torchvision.transforms as transforms
import os, glob
import pandas as pd


def get_optimizer(model: torch.nn.Module, lr: float = 1e-3, weight_decay: float = None, optimizer_type: str = None) -> torch.optim.Optimizer:

    if optimizer_type is None or optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay is not None else 0)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay is not None else 0)
    elif optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay if weight_decay is not None else 0)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    return optimizer


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
        Taken from Beyond Image to Depth Github repository
    """
    # Select only the values that are greater than zero
    mask = gt > 0
    pred = pred[mask]
    gt = gt[mask]

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1=0.0
    if a2 != a2:
        a2=0.0
    if a3 != a3:
        a3=0.0
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel:
        abs_rel=0.0
    if log_10 != log_10:
        log_10=0.0
    if mae != mae:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae


def get_transform(cfg, convert =  False, depth_norm = False):
    # Create list of transform to apply to data
    transform_list = []

    if convert:
        # Convert data to Tensor type
        transform_list += [transforms.ToTensor()]

    if 'resize' in cfg.dataset.preprocess:
        # Resize
        transform_list.append(transforms.Resize((cfg.dataset.images_size,cfg.dataset.images_size)))

    if depth_norm:
        # MinMax depth normalization
        max_depth_dataset = cfg.dataset.max_depth 
        min_depth_dataset = 0.0
        transform_list += [MinMaxNorm(min = min_depth_dataset, max = max_depth_dataset)]

    return transforms.Compose(transform_list)


# Custom transforms 
class MinMaxNorm(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        assert isinstance(min, (float, tuple)) and isinstance(max, (float, tuple))
        self.min = torch.tensor(min)
        self.max = torch.tensor(max)
        
    def forward(self, tensor):
        if tensor.shape[0] == 2:
            norm_tensor_c0 = (tensor[0,...] - self.min[0]) / (self.max[0] - self.min[0])
            norm_tensor_c1 = (tensor[1,...] - self.min[1]) / (self.max[1] - self.min[1])
            norm_tensor = torch.concatenate([norm_tensor_c0.unsqueeze(0), norm_tensor_c1.unsqueeze(0)], dim = 0)

        else:
            norm_tensor = (tensor - self.min) / (self.max - self.min)

        return norm_tensor