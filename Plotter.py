"""
This script is for plotting samples of the
training chips.
"""

from utils.dataset_Oakville_V1 import data, sample
import matplotlib.pyplot as plt

# using 256x256 tensors
plt.plot(data)
plt.title("Image")
plt.show()

plt.plot(sample)
plt.title("Mask")
plt.show()