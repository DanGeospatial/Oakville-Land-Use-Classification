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
import numpy as np
from sklearn.metrics import classification_report

# Import algorithms
from Models.Simple_CNN import Net
from utils.dataset_Oakville_V1 import train_set, test_set, validation_set

print("Using PyTorch version: ", torch.torch_version)
print("With CUDA version: ", torch.cuda_version)


