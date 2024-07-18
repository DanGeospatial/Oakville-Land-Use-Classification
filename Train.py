"""

"""

# Import PyTorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn
import torchvision.transforms as tf
from torch import optim
from torch.utils.data import DataLoader, random_split

# Import other libraries
import os
from pathlib import PureWindowsPath
from sklearn.metrics import classification_report

# Import algorithms
from Models.Simple_CNN import Net
#from utils.dataset import

print("Using PyTorch version: ", torch.torch_version)
print("With CUDA version: ", torch.cuda_version)

# Load file paths
chips = PureWindowsPath('d:/Projects/BuildingExtractor/trainingdata/OakvilleClassificationv1/images')
masks = PureWindowsPath('d:/Projects/BuildingExtractor/trainingdata/OakvilleClassificationv1/labels')

