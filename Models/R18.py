import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def res18 (num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    for params in model.parameters():
        params.requires_grad = True

    model.fc = nn.Linear(512, num_classes)
    return model