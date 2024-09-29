import torch
from Models.Simple_CNN import Net

num_classes = 3
num_bands = 4
path = "I:/LandUseClassification.pth"


model = Net(num_bands, num_classes)
model.load_state_dict(torch.load(path, weights_only=True))
model.eval()