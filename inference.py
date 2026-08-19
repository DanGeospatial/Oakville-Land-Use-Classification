import os
import numpy as np
import torch
from PIL import Image
from torchgeo.models import FarSeg
import matplotlib.pyplot as plt
import rasterio
from rasterio import MemoryFile, merge
from rasterio.plot import show
from torch import device, cuda, autocast
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, stack_samples
from torchgeo.samplers import GridGeoSampler


Image.MAX_IMAGE_PIXELS = None
print("Using PyTorch version: ", torch.__version__)
device = device('cuda' if cuda.is_available() else 'cpu')
print(f"Inferencing on {device}")

image_path = '/mnt/d/LandUse/composite_norm.tif'
output_path = '/mnt/d/LandUse/test_output.tif'
loader_args = dict(num_workers=os.cpu_count(), pin_memory=True)
batch_size = 15
num_classes = 5
chip_stride = 112
size = 256


# Load the model
model = FarSeg(backbone='resnet18', classes=num_classes, backbone_pretrained=True).to(device)
model.load_state_dict(torch.load("/mnt/d/LandUseModel.pth"))
model.eval()

input_drone_image = RasterDataset(image_path)
inference_sampler = GridGeoSampler(input_drone_image, size=size, stride=chip_stride)
inference_set = DataLoader(input_drone_image , batch_size=batch_size, sampler=inference_sampler, collate_fn=stack_samples,
                       **loader_args)


bounds_list = []
image_list = []
crs_list = []

with (torch.inference_mode()):
    for batch in inference_set:
        images = batch["image"]
        bs = images.shape[0]
        images = images.to(device=device)

        with autocast(device.type):
            output = model(images.half())


        for i in range(bs):
            bb = batch["bounds"][i]
            im = output[i].cpu()
            cr = batch["crs"][i]
            bounds_list.append(bb)
            image_list.append(im)
            crs_list.append(cr)


raster_list = []

for i in range(len(image_list)):
    trans = rasterio.transform.from_bounds(west=float(bounds_list[i][0]), north=float(bounds_list[i][3]),
                                           east=float(bounds_list[i][1]), south=float(bounds_list[i][2]), width=256,
                                           height=256)

    pred = torch.argmax(image_list[i], dim=0).numpy().astype(np.uint8)

    profile = {
        'driver': 'GTiff',
        'height': size,
        'width': size,
        'count': 1,  # Number of bands,
        'dtype': np.uint8,
        'crs': crs_list[i],
        'transform': trans
    }
    memfile = MemoryFile()
    rst = memfile.open(**profile)
    rst.write(pred, 1)
    raster_list.append(rst)


mosaic, out_trans = merge.merge(raster_list, method='last')
# show(mosaic, cmap='tab10')
# Update the metadata
out_meta = {"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": crs_list[0],
                 'dtype': np.uint8,
                 'count': 1,
                 }
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(mosaic)

ras = raster_list[15].read()
show(ras, cmap='tab10')
# Convert to segmentation map
segmentation_map = torch.argmax(image_list[14], dim=0).numpy().astype(np.uint8)
# Visualize the segmentation map
plt.imshow(segmentation_map, cmap='tab10', interpolation='none')  # Use a colormap that supports 5 classes (0-4)
# plt.title("Reassembled Segmentation Map")
# plt.colorbar()
# plt.axis('off')
plt.show()
