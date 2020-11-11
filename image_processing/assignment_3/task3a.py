import utils
import skimage
import skimage.morphology as morph
import numpy as np
import matplotlib.pyplot as plt


def remove_noise(im: np.ndarray) -> np.ndarray:
    """
        A function that removes noise in the input image.
        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    # An erosion radius of 7 is the lowest radius which remove all outliers
    erosion_radius = 7
    # A dilation radius of 13 is the lowest radius which fills up the holes
    dilation_radius = 13

    # Since the shrink/grow can have different radius, we compute the difference between them
    # and apply it later as erosion/dilation such that the original shape has the same width/height.
    dilation_erosion_difference = dilation_radius - erosion_radius

    plt.figure(figsize=(16, 6))
    plt_columns = 1
    plt_rows = 4
    plt.subplot(plt_columns, plt_rows, 1)
    plt.title("Before")
    plt.imshow(im)

    im = morph.binary_erosion(im, selem=morph.disk(erosion_radius))

    plt.subplot(plt_columns, plt_rows, 2)
    plt.title("After erosion (shrink)")
    plt.imshow(im)

    im = morph.binary_dilation(im, selem=morph.disk(dilation_radius))

    plt.subplot(plt_columns, plt_rows, 3)
    plt.title("After dilation (grow)")
    plt.imshow(im)

    if(dilation_erosion_difference > 0):
        im = morph.binary_erosion(
            im, selem=morph.disk(dilation_erosion_difference))

        plt.subplot(plt_columns, plt_rows, 4)
        plt.title("After second erosion (shrink)")
        plt.imshow(im)
    elif(dilation_erosion_difference < 0):
        im = morph.binary_dilation(
            im, selem=morph.disk(dilation_erosion_difference))

        plt.subplot(plt_columns, plt_rows, 4)
        plt.title("After second dilation (grow)")
        plt.imshow(im)

    # plt.savefig("image_processed/task3a_pyplot.png")
    # plt.show()
    plt.close()

    return im
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)

    assert im.shape == noise_free_image.shape, "Expected image shape ({}) to be same as resulting image shape ({})".format(
        im.shape, noise_free_image.shape)
    assert noise_free_image.dtype == np.bool, "Expected resulting image dtype to be np.bool. Was: {}".format(
        noise_free_image.dtype)

    noise_free_image = utils.to_uint8(noise_free_image)
    utils.save_im("noisy-filtered.png", noise_free_image)
