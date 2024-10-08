import os
from torch.utils.data import random_split, DataLoader
from utils.DataLoad import Loader

# Load file paths for v1 data
chips = 'I:/OakvilleClassificationv1/images/'
masks = 'I:/OakvilleClassificationv1/labels/'

loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

validation_ratio = 0.20
test_ratio = 0.20
batch_size = 10

dataset = Loader(chips, masks)
data, sample = dataset.__getitem__(0)

# Get counts for each split based on dataset length
test_count, validation_count = int(dataset.__len__() * test_ratio), int(dataset.__len__() * validation_ratio)

# Randomly split the data into train, test, validation
train, validation, test = random_split(dataset, [(dataset.__len__() - (test_count + validation_count)),
                                                 test_count, validation_count
                                                 ])

# Load train, test and validation datasets
train_set = DataLoader(train, batch_size=batch_size, shuffle=True, **loader_args)
test_set = DataLoader(test, batch_size=batch_size, shuffle=True, **loader_args)
validation_set = DataLoader(validation, batch_size=batch_size, shuffle=False, **loader_args)


def getLength():
    return train.__len__()
