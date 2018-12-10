import sys

import pywt

import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
from pyunlocbox import solvers, functions

from PIL import Image

filter_type = None


def gaussian_filter(shape=(3, 3), sigma=0.5):
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp( -(x*x + y*y)/(2. *sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def circular_filter(radius=5):
    cradius = np.ceil(radius)
    x, y = np.ogrid[-cradius:cradius+1, -cradius:cradius+1]
    disk = x**2 + y**2 <= radius**2
    return disk.astype(float)


def run(pathname, filter_type='gaussian', wavelet_type='haar'):

    wavelet = pywt.Wavelet(wavelet_type)

    def compute_approx_single_depth(alpha, coeff_slice, d):
        coeffs = pywt.array_to_coeffs(alpha, coeff_slice, 'wavedec2')
        X = pywt.waverec2(coeffs, wavelet=wavelet)
        k = None
        if filter_type == 'gaussian':
            k = gaussian_filter(sigma=d)
        else:
            k = circular_filter(radius=d)
        Y = signal.convolve2d(X, k)
        return Y

    img = misc.imread(pathname)
    print(img)
    print(type(img))
    img = Image.open(pathname)
    print(img)
    print(type(img))
    img = np.array(img)
    print(img)
    print(type(img))
    img_no_blurry = Image.open(pathname.replace('blurry', 'original'))

    coeffs = pywt.wavedec2(img_no_blurry, wavelet)
    alpha, coeffs_slice = pywt.coeffs_to_array(coeffs)
    Y = compute_approx_single_depth(alpha, coeffs_slice, 3)
    # img2 = pywt.waverec2(coeffs, wavelet)

    solver = solvers.forward_backward()
    n = len(img)
    x0 = np.zeros(n)


    def error_term():
        return 0


    def alpha_sparsity(img):
        return 0


    def mask_sparsity(img):
        return 0

    ret = solvers.solve([], x0, solver, rtol=1e-4, maxit=300)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(80, 50))
    plt.gray()

    for a in (ax[0], ax[1], ax[2]):
        a.axis('off')

    ax[0].imshow(img)
    ax[0].set_title('Blurry Data')

    ax[1].imshow(Y)
    ax[1].set_title('reconstruct')

    #ax[2].imshow(img_noblurry)
    #ax[2].set_title('original')

    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 3:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2:
        run(sys.argv[1], sys.argv[2])
    else:
        run(sys.argv[1])
