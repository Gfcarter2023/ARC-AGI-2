from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import warnings
import ast
from tqdm import tqdm
import random

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns
warnings.filterwarnings('ignore')


class ConditionalBatchNorm2d(nn.Module):
    """
    Conditional Batch Normalization where gamma and beta are predicted from a conditioning embedding.
    Used in the OutputDecoder to make normalization adaptive to the inferred rule.
    """
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False) # affine=False means BN learns no scale/shift
        self.gamma_beta_mlp = nn.Linear(cond_dim, 2 * num_features) # MLP to predict scale and shift

    def forward(self, x, cond_embedding):
        out = self.bn(x)
        # Predict gamma and beta from the conditioning embedding
        gamma, beta = self.gamma_beta_mlp(cond_embedding).chunk(2, dim=1)
        # Reshape for broadcasting across spatial dimensions
        gamma = gamma.view(gamma.size(0), self.num_features, 1, 1)
        beta = beta.view(beta.size(0), self.num_features, 1, 1)
        # Apply adaptive scale and shift
        return out * (1 + gamma) + beta

class ResidualBlock(nn.Module):
    """
    A simple Residual Block for the ARCNeuralNetwork (encoder).
    Helps in training deeper networks and preventing vanishing gradients.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        # If input/output channels or stride differ, a 1x1 convolution is needed for shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x))) # Using GELU
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # Add shortcut connection
        out = F.gelu(out) # Using GELU
        return out