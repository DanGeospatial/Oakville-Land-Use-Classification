from pathlib import PureWindowsPath
from torch.utils.data import random_split

# Load file paths for v1 data
chips = PureWindowsPath('I:/OakvilleClassificationv1/images')
masks = PureWindowsPath('I:/OakvilleClassificationv1/labels')

train_ratio = 0.60
validation_ratio = 0.20
test_ratio = 0.20

# train is now 60% of the entire data set


