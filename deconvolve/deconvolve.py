import sys

import pywt

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pyunlocbox import solvers, functions

from PIL import Image


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


def run(pathname, filter_type='gaussian', patch_size=64, wavelet_type='haar', lambda_s=0.1, initial_depth=3):

    wavelet = pywt.Wavelet(wavelet_type)

    img = Image.open(pathname)
    img = np.array(img)
    Y = img  # objective value

    def build_mask():
        num_patch_x, num_patch_y = np.ceil(float(img.shape[0])/patch_size), np.ceil(float(img.shape[1])/patch_size)
        img_x, img_y = img.shape
        mask = np.zeros((num_patch_x, num_patch_y, img_x, img_y))
        for i in range(0, num_patch_x):
            for j in range(0, num_patch_y):
                lim_x_min, lim_x_max = i * patch_size, min((i+1) * patch_size, img_x)
                lim_y_min, lim_y_max = j * patch_size, min((j+1) * patch_size, img_y)
                mask[i][j][lim_x_min:lim_x_max][lim_y_min:lim_y_max] = 1
        return mask.reshape(num_patch_x * num_patch_y, img_x * img_y)

    # mask = build_mask()
    mask = np.ones((1, img.size))

    def compute_approx_single_depth(alpha, coeff_slice, d):
        """alpha should be 2-D"""
        coeffs = pywt.array_to_coeffs(alpha, coeff_slice, 'wavedec2')
        X = pywt.waverec2(coeffs, wavelet=wavelet)
        k = None
        if filter_type == 'gaussian':
            k = gaussian_filter(sigma=d)
        else:
            k = circular_filter(radius=d)
        Y = signal.convolve2d(X, k)
        return Y

    # The following is a test
    img_no_blurry = np.array(Image.open(pathname.replace('blurry', 'original')))
    coeffs = pywt.wavedec2(img_no_blurry, wavelet)
    alpha, coeffs_slice = pywt.coeffs_to_array(coeffs)
    X_hat_from_no_blurry = compute_approx_single_depth(alpha, coeffs_slice, 3)

    dumb_coeffs = pywt.wavedec2(img, wavelet)
    dumb_alpha, coeffs_slice = pywt.coeffs_to_array(dumb_coeffs)
    alpha_size = dumb_alpha.size
    num_depths = 3
    num_segmentations = mask.shape[0]

    n = alpha_size + num_segmentations
    x0 = np.zeros(n)
    x0[alpha_size:] = initial_depth

    def compute_X_hat(x):
        alpha = x[0:alpha_size]
        depths_params = x[alpha_size:alpha_size+num_depths]
        X_hat = compute_approx_single_depth(alpha, coeffs_slice, depths_params[0])
        return X_hat

    def error_term(x):
        X_hat = compute_X_hat(x)
        return np.sum((X_hat - Y)**2)

    """
    def error_term_grad(x):
        alpha = x[0:alpha_size]
        depths_params = x[alpha_size:alpha_size+num_depths]
        X_hat = compute_approx_single_depth(alpha, coeffs_slice, depths_params[0])
    """

    class MyNorm2(functions.norm_l2):
        def __init__(self, **kwargs):
            super(MyNorm2, self).__init__(**kwargs)
            self.A = compute_X_hat

        def eval(self, x):
            return error_term(x)

        def grad(self, x):
            tmp = super(MyNorm2, self).grad(x)
            tmp[alpha_size:] = 0 # Otherwise we will move out of the coefficient
            return tmp

    error_func = MyNorm2(y=Y)
    error_func.prox = None # Remove it
    # error_func.cap = lambda x: ["eval"]
    # error_func.eval = lambda x: error_term(x)
    # TODO add grad and prox

    def alpha_sparsity1(x):
        alpha = x[0:alpha_size]
        return np.sum(np.abs(alpha))

    class MyNorm1(functions.norm_l1):
        def __init__(self, **kwargs):
            super(MyNorm1, self).__init__(**kwargs)

        def eval(self, x):
            return alpha_sparsity1(x)

        def prox(self, x, T):
            alpha = x[0:alpha_size]
            return super(MyNorm1, self).prox(alpha, T)

    alpha_sparsity_func = MyNorm1(lambda_=lambda_s)
    # alpha_sparsity_func.cap = lambda x: ["eval"]
    # alpha_sparsity_func.eval = lambda x: lambda_s * alpha_sparsity1(x)

    # alpha_sparsity2 = functions.norm_l1(lambda_=lambda_s)

    solver = solvers.generalized_forward_backward()
    ret = solvers.solve([error_func, alpha_sparsity_func], x0, solver, rtol=1e-4, maxit=300, verbosity='ALL')
    img_reconstructed = ret['sol']
    print("Number of iterations".format(ret['niter']))

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(50, 50))
    plt.gray()

    for a in (ax[0], ax[1], ax[2], ax[3]):
        a.axis('off')

    ax[0].imshow(img)
    ax[0].set_title('Blurry Data')

    ax[1].imshow(X_hat_from_no_blurry)
    ax[1].set_title('Blurried image from the no-blurry image')

    ax[2].imshow(img_no_blurry)
    ax[2].set_title('original')

    ax[3].imshow(img_reconstructed)
    ax[3].set_title('reconstructed image')

    fig.subplots_adjust(wspace=0.02, hspace=0.2,
                        top=0.9, bottom=0.05, left=0, right=1)
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) > 5:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif len(sys.argv) > 4:
        run(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 3:
        run(sys.argv[1], sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2:
        run(sys.argv[1], sys.argv[2])
    else:
        run(sys.argv[1])
