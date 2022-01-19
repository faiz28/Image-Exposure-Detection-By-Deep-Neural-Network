import sys
import numpy as np
import skimage.color
import skimage.io
from matplotlib import pyplot as plt


image = skimage.io.imread(fname='img1.JPG')

# display the image
# skimage.io.imshow(image)
# histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

# plt.figure()
# plt.title("Grayscale Histogram")
# plt.xlabel("grayscale value")
# plt.ylabel("pixels")
# plt.xlim([0.0, 1.0])  # <- named arguments do not work here

# plt.plot(bin_edges[0:-1], histogram)  # <- or here
# plt.show()
# plt.close()
# plt.hist(image.flatten(), bins=256, range=(0, 1))
# plt.show()

# tuple to select colors of each channel line
colors = ("red", "green", "blue")
channel_ids = (0, 1, 2)

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for channel_id, c in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        image[:, :, channel_id], bins=256, range=(0, 256)
    )
    plt.plot(bin_edges[0:-1], histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")

plt.show()
