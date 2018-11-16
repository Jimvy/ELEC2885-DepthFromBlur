import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import convolve2d as conv2

from skimage import color, data, restoration

img = Image.open("DFB_artificial_dataset/im0_blurry.bmp")

# Restore Image using Richardson-Lucy algorithm
deconvolved = img

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 5))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
       a.axis('off')

ax[0].imshow(img)
ax[0].set_title('Blurry Data')

ax[2].imshow(deconvolved)
ax[2].set_title('Restoration')


fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()
