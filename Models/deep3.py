import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

def deeplab(num_classes, num_bands, device):
    model = deeplabv3_resnet50(num_classes=num_classes)

    for params in model.parameters():
        params.requires_grad = True

    model.backbone.conv1 = nn.Conv2d(num_bands, 64, 7, 2, 3, bias=False)
    model.to(device)
    return model
