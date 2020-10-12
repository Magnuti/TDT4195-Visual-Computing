import matplotlib.pyplot as plt
import pathlib
from utils import read_im, save_im
import numpy as np

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)


im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)
# plt.show()


def greyscale(im):
    """ Converts an RGB image to greyscale

    Args:
        im ([type]): [np.array of shape [H, W, 3]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    # grey = 0.212R + 0.7152G + 0.0722B
    return im.dot([0.212, 0.7152, 0.0722])

    # Without weights:
    # return np.sum(a=im, axis=2)


im_greyscale = greyscale(im)
save_im(output_dir.joinpath("lake_greyscale.jpg"), im_greyscale, cmap="gray")
plt.imshow(im_greyscale, cmap="gray")
# plt.show()

# print("Image range: {}-{} ".format(im.min(), im.max()))


def inverse(im):
    """ Finds the inverse of the greyscale image

    Args:
        im ([type]): [np.array of shape [H, W]]

    Returns:
        im ([type]): [np.array of shape [H, W]]
    """

    return 1.0 - im


im_inversed = inverse(im_greyscale)
save_im(output_dir.joinpath("lake_greyscale_inversed.jpg"),
        im_inversed, cmap="gray")
plt.imshow(im_inversed, cmap="gray")
# plt.show()
