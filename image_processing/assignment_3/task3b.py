import utils
import skimage
import skimage.morphology
import numpy as np
from task3a import remove_noise


def distance_transform(im: np.ndarray) -> np.ndarray:
    """
        A function that computes the distance to the closest boundary pixel.

        args:
            im: np.ndarray of shape (H, W) with boolean values (dtype=np.bool)
        return:
            (np.ndarray) of shape (H, W). dtype=np.int32
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    assert im.dtype == np.bool
    structuring_element = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=bool)
    result = im.astype(np.int32)
    found = np.zeros(im.shape).astype(bool)

    # First iterator, find initial black values
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            if(not im[row, col]):
                result[row, col] = 0
                found[row, col] = True

    counter = 0
    while(im.max()):
        counter += 1
        im = skimage.morphology.binary_erosion(im, structuring_element)
        for row in range(im.shape[0]):
            for col in range(im.shape[1]):
                if(not im[row, col] and not found[row, col]):
                    result[row, col] = counter
                    found[row, col] = True

    return result
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    im = utils.read_image("noisy.png")
    binary_image = (im != 0)
    noise_free_image = remove_noise(binary_image)
    distance = distance_transform(noise_free_image)

    assert im.shape == distance.shape, "Expected image shape ({}) to be same as resulting image shape ({})".format(
        im.shape, distance.shape)
    assert distance.dtype == np.int32, "Expected resulting image dtype to be np.int32. Was: {}".format(
        distance.dtype)

    distance = utils.to_uint8(distance)
    utils.save_im("noisy-distance.png", distance)
