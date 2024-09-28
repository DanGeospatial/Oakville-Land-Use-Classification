import os
import rasterio
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
        image = ToTensor()(image.read())

        image = MinMaxScaler().fit_transform(image)

        label_file = self.labels[item]
        label = rasterio.open((self.label_folder + label_file))
        label = ToTensor()(image.read())

        return image, label

    def __len__(self):
        return len(self.images)



