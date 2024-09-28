from pathlib import PureWindowsPath
from torch.utils.data import DataLoader, random_split

# Load file paths for v1 data
chips = PureWindowsPath('d:/Projects/BuildingExtractor/trainingdata/OakvilleClassificationv1/images')
masks = PureWindowsPath('d:/Projects/BuildingExtractor/trainingdata/OakvilleClassificationv1/labels')


