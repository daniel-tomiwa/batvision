import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import functools
import matplotlib.pyplot as plt
import pandas as pd

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'group':
        def norm_layer(x): return nn.GroupNorm(num_groups=32, num_channels=x, affine=True)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1 or classname.find('GroupNorm') != -1:  # BatchNorm and GroupNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


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
    

def visualize_predictions(combined_df: pd.DataFrame, num_samples: int = 5, random_seed: int = 42, show: bool = True) -> plt.Figure:

    n_samples = min(num_samples, len(combined_df))
    samples = combined_df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)

    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))

    for i in range(n_samples):
        instance = samples.iloc[i]

        gt_depth = instance['gt_images'][0, :, :]
        pred_depth = instance['pred_imgs'][0, :, :]
        original_image = instance['input_images']


        axes[i, 0].imshow(original_image)
        axes[i, 0].set_title('RGB Image')
        axes[i, 0].axis('off')

        im1 = axes[i, 1].imshow(gt_depth, cmap='plasma')
        axes[i, 1].set_title('Ground Truth Depth')
        axes[i, 1].axis('off')
        fig.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

        im2 = axes[i, 2].imshow(pred_depth, cmap='plasma')
        axes[i, 2].set_title('Predicted Depth')
        axes[i, 2].axis('off')
        fig.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if show:
        plt.show()

    return fig