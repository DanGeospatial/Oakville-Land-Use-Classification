"""
This version has fixes for creating tiles and loading satellite images
Use torchgeo to streamline this task
"""
import os
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler
from torch.utils.data import DataLoader


raster_location = '/mnt/d/TWNOakvilleJuly22_23_psscene_analytic_sr_udm2/image/'
mask_location = '/mnt/d/OakvilleMask/masks.tif'

loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

validation_ratio = 0.20
batch_size = 20

ps_image = RasterDataset(raster_location)
oak_mask = RasterDataset(mask_location)

oak_mask.is_image = False

# get the intersection of two datasets because ps_image and oak_mask cover slightly different areas
dataset = ps_image & oak_mask

# sample from ps_image at 224 pixel tiles
train_sampler = GridGeoSampler(ps_image, size=224, stride=64)
# validation_sampler = GridGeoSampler(dataset , size=256, stride=64)

# Load train, test and validation datasets
train_set = DataLoader(dataset , batch_size=batch_size, sampler=train_sampler, collate_fn=stack_samples,
                       **loader_args)
# validation_set = DataLoader(dataset, batch_size=batch_size, sampler=validation_sampler, collate_fn=stack_samples,
# **loader_args)

"""
for batch in train_set:
    images = batch["image"][0]
    masks = batch["mask"][0]

    print(masks)
    break
"""

def getLength():
    return dataset.__len__()
