import matplotlib.pyplot as plt
import pathlib
import numpy as np
from utils import read_im, save_im, normalize

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)

im = read_im(pathlib.Path("images", "lake.jpg"))
plt.imshow(im)


def convolve_im(im, kernel,
                ):
    """ A function that convolves im with kernel

    Args:
        im ([type]): [np.array of shape [H, W, 3]]
        kernel ([type]): [np.array of shape [K, K]]

    Returns:
        [type]: [np.array of shape [H, W, 3]. should be same as im]
    """
    assert len(im.shape) == 3

    image_height = im.shape[0]
    image_width = im.shape[1]
    kernel_dim_divided = kernel.shape[0] // 2

    computed = np.zeros(shape=(image_height, image_width, im.shape[2]))

    # Different loop-orders were tried out, but this one seemed
    # to yield the lowest running time
    for channel_index in range(im.shape[2]):  # RGB channel
        for h_index in range(image_height):
            for w_index in range(image_width):
                pixel_value = 0.0
                for ki in range(-kernel_dim_divided, kernel_dim_divided + 1):
                    h_coor = h_index - ki
                    if h_coor < 0 or h_coor >= image_height:
                        continue
                    for kj in range(-kernel_dim_divided, kernel_dim_divided + 1):
                        # Minus since we are using applying convolution
                        w_coor = w_index - kj
                        if w_coor < 0 or w_coor >= image_width:
                            continue
                        pixel_value += kernel[kernel_dim_divided + ki,
                                              kernel_dim_divided + kj] * im[h_coor, w_coor, channel_index]
                computed[h_index, w_index, channel_index] = pixel_value

    return computed


# Define the convolutional kernels

# The Gaussian blur 5x5
h_b = 1 / 256 * np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
])

# Sobel kernel vertical edge detection
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

# Convolve images
im_smoothed = convolve_im(im.copy(), h_b)
save_im(output_dir.joinpath("im_smoothed.jpg"), im_smoothed)

im_sobel = convolve_im(im, sobel_x)
save_im(output_dir.joinpath("im_sobel.jpg"), im_sobel)

# DO NOT CHANGE. Checking that your function returns as expected
assert isinstance(
    im_smoothed, np.ndarray),     f"Your convolve function has to return a np.array. " + f"Was: {type(im_smoothed)}"
assert im_smoothed.shape == im.shape,     f"Expected smoothed im ({im_smoothed.shape}" + \
    f"to have same shape as im ({im.shape})"
assert im_sobel.shape == im.shape,     f"Expected smoothed im ({im_sobel.shape}" + \
    f"to have same shape as im ({im.shape})"


plt.subplot(1, 2, 1)
plt.imshow(normalize(im_smoothed))
# plt.show()

plt.subplot(1, 2, 2)
plt.imshow(normalize(im_sobel))
# plt.show()
