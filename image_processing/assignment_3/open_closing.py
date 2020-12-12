import utils
import skimage
import skimage.morphology as mo
import numpy as np
import matplotlib.pyplot as plt


def open_close(im):
    structuring_element = mo.disk(3)

    plt.figure(figsize=(16, 5))
    plt_columns = 4
    plt_rows = 1

    plt.subplot(plt_rows, plt_columns, 1)
    plt.imshow(im)
    plt.title("Original")

    open_im = mo.binary_opening(im, selem=structuring_element)
    plt.subplot(plt_rows, plt_columns, 2)
    plt.imshow(open_im)
    plt.title("After opening")

    close_im = mo.binary_closing(im, selem=structuring_element)
    plt.subplot(plt_rows, plt_columns, 3)
    plt.imshow(close_im)
    plt.title("After closing")

    open_close_im = im.copy()
    for i in range(20):
        open_close_im = mo.binary_opening(open_close_im, selem=structuring_element)
        open_close_im = mo.binary_closing(open_close_im, selem=structuring_element)

    plt.subplot(plt_rows, plt_columns, 4)
    plt.imshow(open_close_im)
    plt.title("Multiple open close")

    plt.savefig("image_processed/open_closing.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    open_close(binary_image)
