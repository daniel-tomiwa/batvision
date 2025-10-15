import torch
import torch.nn as nn
from omegaconf import DictConfig


class ASTModel(nn.Module):

    def __init__(self, cfg: DictConfig):
        super().__init__()

        pass

    def forward(self, x):
        return self.model(x)