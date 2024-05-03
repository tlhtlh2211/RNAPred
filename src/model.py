from torch import nn
from torch.nn.functional import cross_entropy
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import math

class Redisual(nn.Module):
    def __init__(
        self,
        train_len=0,
        embedding_dim=4,
        device="cpu",
        negative_weight=0.1,
        lr=1e-4,
        loss_l1=0,
        loss_beta=0,
        scheduler="none",
        verbose=True,
        interaction_prior=False,
        output_th=0.5,
        **kwargs
    ):
        super().__init__()
        

class ResidualLayer1D(nn.Module):
    '''
    This class defines a residual layer for 1D data
    '''
    def __init__(self, 
                 filters,
                 rb_factor,
                 kernel_size, 
                 dilation):
        
        super().__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(filters),
            nn.ReLU(),
            nn.Conv1d(filters,
                      math.floor(rb_factor * filters),
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
            nn.BatchNorm1d(math.floor(rb_factor * filters)),
            nn.ReLU(),
            nn.Conv1d(math.floor(rb_factor * filters), 
                      filters,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)
    
class ResidualLayer2D(nn.Module):
    '''
    This class defines a residual layer for 2D data
    '''
    def __init__(self, 
                 filters,
                 factors,
                 kernel_size, 
                 dilation):
        
        super().__init__()

        self.layer = nn.Sequential(
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters,
                      factors,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
            nn.BatchNorm2d(factors),
            nn.ReLU(),
            nn.Conv2d(factors, 
                      filters,
                      kernel_size, 
                      dilation=dilation, 
                      padding="same"),
        )

    def forward(self, x):
        return x + self.layer(x)