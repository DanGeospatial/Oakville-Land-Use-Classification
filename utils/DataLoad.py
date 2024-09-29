import os
import rasterio
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from sklearn.preprocessing import MinMaxScaler


class Loader(Dataset):
    def __init__(self, image_folder, label_folder):
        self.image_folder = image_folder
        self.images = os.listdir(image_folder)
        self.label_folder = label_folder
        self.labels = os.listdir(label_folder)

    def __getitem__(self, item):
        image_file = self.images[item]

        image = rasterio.open((self.image_folder + image_file))
        image = np.array(image.read())
        image = MinMaxScaler().fit_transform(image)
        image = ToTensor()(image)

        label_file = self.labels[item]
        label = rasterio.open((self.label_folder + label_file))
        label = np.array(label.read())
        label = ToTensor()(label)

        return image, label

    def __len__(self):
        return len(self.images)



