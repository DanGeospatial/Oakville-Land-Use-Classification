"""
This version has fixes for creating tiles and loading satellite images
Use torchgeo to streamline this task
"""
import os
from torchgeo.datasets import RasterDataset, VectorDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torch.utils.data import random_split, DataLoader


raster_location = 'I:/image/'
vector_location = 'I:/mask/'

loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

validation_ratio = 0.20
test_ratio = 0.20
batch_size = 20

ps_image = RasterDataset(raster_location)
oak_mask = VectorDataset(vector_location, label_name="classvalue")

# get the intersection of two datasets because ps_image and oak_mask cover slightly different areas
dataset = ps_image & oak_mask

# Get counts for each split based on dataset length
test_count, validation_count = int(dataset.__len__() * test_ratio), int(dataset.__len__() * validation_ratio)

# Randomly split the data into train, test, validation
train, validation, test = random_split(dataset, [(dataset.__len__() - (test_count + validation_count)),
                                                 test_count, validation_count
                                                 ])

# sample from ps_image at 256 pixel tiles
train_sampler = GridGeoSampler(train, size=256, stride=64)
test_sampler = GridGeoSampler(test, size=256, stride=64)
valid_sampler = GridGeoSampler(validation, size=256, stride=64)

# Load train, test and validation datasets
train_set = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=stack_samples, **loader_args)
validation_set = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, collate_fn=stack_samples, **loader_args)


def getLength():
    return train.__len__()
