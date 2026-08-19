import rasterio
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples, splits
from torchgeo.samplers import RandomGeoSampler


image_path = '/mnt/d/LandUse/composite_RGB.tif'
image_norm = '/mnt/d/LandUse/composite_norm.tif'
mask_path = '/mnt/d/LandUse/mask_v1.tif'
batch_size = 10
slide = 224
size = 224
loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)

def normalize(x: np.ndarray, percentile: int = 100) -> np.ndarray:
    """Min/max normalize to [0, 1] range given a percentile."""
    c, h, w = x.shape
    x = x.reshape(c, -1)
    min = np.percentile(x, 100 - percentile, axis=-1)[:, None, None]
    max = np.percentile(x, percentile, axis=-1)[:, None, None]
    x = x.reshape(c, h, w)
    x = np.clip(x, min, max)
    return (x - min) / (max - min)

ds1 = rasterio.open(image_path)
x1 = ds1.read()
x1 = normalize(x1, percentile=99)

transform = ds1.transform

out_meta = {"driver": "GTiff",
                 "height": x1.shape[1],
                 "width": x1.shape[2],
                 "crs": 'EPSG:32617',
                 "transform": transform,
                 'dtype': np.float32,
                 'count': 3,
                 }
with rasterio.open(image_norm, "w", **out_meta) as dest:
    dest.write(x1)
ds1.close()

input_drone_image = RasterDataset(image_norm)
drone_mask = RasterDataset(mask_path)
drone_mask.is_image = False

dataset = input_drone_image & drone_mask
generator = torch.Generator().manual_seed(32)
(train, val) = splits.random_grid_cell_assignment(dataset, [0.8, 0.2], generator=generator)

train_sampler = RandomGeoSampler(train, size=size)
train_set = DataLoader(train, batch_size=batch_size, sampler=train_sampler, collate_fn=stack_samples,
                       **loader_args)
val_sampler = RandomGeoSampler(val, size=size)
val_set = DataLoader(val, batch_size=batch_size, sampler=val_sampler, collate_fn=stack_samples,
                     **loader_args)

print(f'Train Size: {len(train_set)}')
print(f'Val Size: {len(val_set)}')
