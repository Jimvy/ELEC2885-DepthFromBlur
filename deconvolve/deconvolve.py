import pywt

import numpy as np
import matplotlib.pyplot as plt
from pyunlocbox import solvers

from PIL import Image

img = Image.open("DFB_artificial_dataset/im0_blurry.bmp")

# Restore Image using Richardson-Lucy algorithm
coeffs = pywt.wavedec2(img, 'db1')
img2 = pywt.waverec2(coeffs, 'db1')

solver = solvers.forward_backward()
n = len(img)
x0 = np.zeros(n)


def f1(img):
    return 0


def f2(img):
    return 0


def f3(img):
    return 0


ret = solvers.solve([f1, f2, f3], x0, solver, rtol=1e-4, maxit=300)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(80, 50))
plt.gray()

for a in (ax[0], ax[1], ax[2]):
    a.axis('off')

ax[0].imshow(img)
ax[0].set_title('Blurry Data')

ax[1].imshow(img2)
ax[1].set_title('reconstruct')

fig.subplots_adjust(wspace=0.02, hspace=0.2,
                    top=0.9, bottom=0.05, left=0, right=1)
plt.show()
