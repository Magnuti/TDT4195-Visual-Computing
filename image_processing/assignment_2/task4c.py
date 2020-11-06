import skimage
import skimage.io
import skimage.transform
import os
import numpy as np
import utils
import matplotlib.pyplot as plt


def magnitude(fft_im):
    real = fft_im.real
    imag = fft_im.imag
    return np.sqrt(real**2 + imag**2)


if __name__ == "__main__":
    # DO NOT CHANGE
    impath = os.path.join("images", "noisy_moon.png")
    im = utils.read_im(impath)

    # Since the spikes are on a simple horizontal line in the center around [265:275], we
    # can simply craete a kernel which removes theese values. Non-shifted, [265:275] becomes [0:10]
    # Note that the middle area should be unaffected.

    kernel = np.ones(im.shape)
    spike_height = 4  # Lines are barely visible at < 4
    spike_height = spike_height // 2
    center_width = 28  # Lines are visible for > 28

    for y in range(- spike_height, spike_height):
        for x in range(center_width, im.shape[1] // 2):
            kernel[y][x] = 0
        for x in range(im.shape[1] // 2, im.shape[1] - center_width):
            kernel[y][x] = 0

    # START YOUR CODE HERE ### (You can change anything inside this block)
    fft_im = np.fft.fft2(im)

    # Note that this is not matrix multiplication, only point-vise multiplication
    # For matrix multiplication, use np.matmul(..)
    fft_im_filtered = fft_im * kernel

    inversed_im = np.fft.ifft2(fft_im_filtered).real

    # Visualization start, don't calculate from here
    fft_im = np.fft.fftshift(fft_im)
    fft_im = magnitude(fft_im)
    fft_im = np.log(fft_im + 1)

    fft_im_filtered = np.fft.fftshift(fft_im_filtered)
    fft_im_filtered = magnitude(fft_im_filtered)
    fft_im_filtered = np.log(fft_im_filtered + 1)

    kernel = np.fft.fftshift(kernel)

    plt.figure(figsize=(16, 8))
    plt_rows = 5

    plt.subplot(1, plt_rows, 1)
    plt.imshow(im, cmap="gray")
    plt.title("Original image")

    plt.subplot(1, plt_rows, 2)
    plt.imshow(fft_im, cmap="gray")
    plt.title("FFT image shifted")

    plt.subplot(1, plt_rows, 3)
    plt.imshow(kernel, cmap="gray")
    plt.title("Kernel")

    plt.subplot(1, plt_rows, 4)
    plt.imshow(fft_im_filtered, cmap="gray")
    plt.title("FFT image filtered")

    plt.subplot(1, plt_rows, 5)
    plt.imshow(inversed_im, cmap="gray")
    plt.title("Inversed image")

    plt.savefig(utils.image_output_dir.joinpath("task4c_full.png"))
    # plt.show()

    # END YOUR CODE HERE ###
    utils.save_im("moon_filtered.png", utils.normalize(inversed_im))
