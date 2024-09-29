"""
Train the land use dataset with 3 classes and 4 image bands
"""

# Import PyTorch libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn
import torchvision.transforms as tf
from torch import device, cuda
from torch.optim import Optimizer
from torch.utils.data import DataLoader, random_split

# Import other libraries
import os
import numpy as np
from sklearn.metrics import classification_report

# Import algorithms
from Models.Simple_CNN import Net
from utils.dataset_Oakville_V1 import train_set, test_set, validation_set

num_classes = 3
num_bands = 4
epochs = 40

for epoch in range(epochs):
    for images, masks in train_set:
        Optimizer.zero_grad()
        images, masks = images.cuda(), masks.cuda()
        predictions = Net.train(images)


if __name__ == '__main__':
    print("Using PyTorch version: ", torch.torch_version)
    print("With CUDA version: ", torch.cuda_version)

    device = device('cuda' if cuda.is_available() else 'cpu')
    print(f"Training on {device}")

    network = Net(num_bands, num_classes)
