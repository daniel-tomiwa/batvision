import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange, reduce
from functools import partial

# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        # compute the mean and variance for each output channel across all spatial dimensions
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class DoubleConv(nn.Module):
    # Initialize the DoubleConv module
    def __init__(self, in_c: int, out_c: int, mid_c=None, residual=False):
        super().__init__()
        self.residual = residual  # Whether to use residual connections
        if not mid_c:
            mid_c = out_c  # If mid_c is not provided, use out_c
        # Define a sequence of operations: two convolutional layers with normalization and activation
        self.double_conv = nn.Sequential(
            WeightStandardizedConv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_c),
            nn.SiLU(),
            WeightStandardizedConv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_c)
        )

    # Forward pass through the module
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))  # Apply residual connection if enabled
        else:
            return self.double_conv(x)  # Otherwise, just pass through the double convolution

class Down(nn.Module):
    # Initialize the Down module for downsampling
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        # Define a sequence of operations: max pooling followed by two DoubleConv modules
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c, out_c)
        )

    # Forward pass through the module
    def forward(self, x):
        x = self.maxpool_conv(x)  # Apply max pooling and convolutions
        return x

class Up(nn.Module):
    # Initialize the Up module for upsampling
    def __init__(self, in_c, out_c):
        super().__init__()
        # Define a sequence of operations: upsampling followed by two DoubleConv modules
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(in_c, out_c, mid_c=in_c//2, residual=False)
        )

    # Forward pass through the module
    def forward(self, x):
        return self.upconv(x)  # Apply upsampling and convolutions
    
class MyUNet(nn.Module):
    def __init__(self, dim:int, init_dim:int=None, out_dim:int=None, dim_mults:tuple=(1, 2, 4, 8), channels:int=1):

        """
        A U-Net architecture for image processing tasks.

        Parameters:
            dim (int): Base dimension for the model.
            init_dim (int, optional): Initial dimension for the first convolutional layer. Defaults to dim if not provided.
            out_dim (int, optional): Output dimension for the final layer.
            dim_mults (tuple, optional): Multiplicative factors for the dimensions at each layer. Defaults to (1, 2, 4, 8).
            channels (int, optional): Number of input channels. Defaults to 1.
        """

        super().__init__()

        self.channels = channels

        init_dim = default(init_dim, dim) # Default initial dimension if not provided
        self.init_conv = nn.Conv2d(self.channels, init_dim, kernel_size=3, padding=1)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)] # Calculate dimensions for each layer

        in_out = list(zip(dims[:-1], dims[1:]))  # Create pairs of input and output dimensions for each layer

        self.downs = nn.ModuleList([])  # List to hold downsampling layers
        self.ups = nn.ModuleList([])    # List to hold upsampling layers

        for ind,  (dim_in, dim_out) in enumerate(in_out):

            is_last = ind >= (len(in_out) - 1)  # Check if it's the last layer

            self.downs.append(
                nn.ModuleList([
                    DoubleConv(dim_in, dim_in, residual=True),
                    Down(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
                ])
            )

        # Middle layers
        mid_dim = dims[-1] # The last element in dims represents the middle dimension
        self.mid_block1 = DoubleConv(mid_dim, mid_dim, residual=True)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):

            is_last = ind == (len(in_out) - 1)  # Check if it's the last layer

            self.ups.append(
                nn.ModuleList([
                    DoubleConv(dim_out + dim_in, dim_out),
                    Up(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
                ])
            )

        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)

        self.final_conv = nn.Conv2d(init_dim, self.out_dim, kernel_size=1)

    def forward(self, x):

        # Pass the input through the initial convolutional layer
        x = self.init_conv(x)

        h = []

        # Downsampling path
        for block, downsample in self.downs:
            x = block(x)  # Apply the first DoubleConv block
            h.append(x)  # Store the output for skip connections
            x = downsample(x)  # Apply downsampling

        # Middle layers
        x = self.mid_block1(x)  # First middle DoubleConv block

        # Upsampling path
        for block, upsample in self.ups:

            x = torch.cat((x, h.pop()), dim=1)  # Concatenate with the corresponding skip connection
            x = block(x)  # Apply the first DoubleConv block
            x = upsample(x)  # Apply upsampling

        return self.final_conv(x)  # Final output convolution