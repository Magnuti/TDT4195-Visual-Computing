import utils
import numpy as np
import matplotlib.pyplot as plt


def region_growing(im: np.ndarray, seed_points: list, T: int) -> np.ndarray:
    """
        A region growing algorithm that segments an image into 1 or 0 (True or False).
        Finds candidate pixels with a Moore-neighborhood (8-connectedness). 
        Uses pixel intensity thresholding with the threshold T as the homogeneity criteria.
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
            seed_points: list of list containing seed points (row, col). Ex:
                [[row1, col1], [row2, col2], ...]
            T: integer value defining the threshold to used for the homogeneity criteria.
        return:
            (np.ndarray) of shape (H, W). dtype=np.bool
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions

    # TODO handle out of bounds on edges?

    segmented = np.zeros_like(im).astype(bool)
    im = im.astype(float)

    def visit_pixel(seed_point_value, row, col):
        for i in range(-1, 2):
            for j in range(-1, 2):
                # Skip the calling pixel
                if(i == 0 and j == 0):
                    continue

                # Skip already found pixels
                if(not segmented[row + i, col + j]):
                    # Calculate if this pixel should be taken into account or not
                    if(abs(im[row + i, col + j] - seed_point_value) < T):
                        segmented[row + i, col + j] = True
                        visit_pixel(seed_point_value, row + i, col + j)

    for row, col in seed_points:
        segmented[row, col] = True
        visit_pixel(im[row, col], row, col)

    # plt.figure()
    # plt.imshow(segmented)
    # plt.show()

    return segmented
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    im = utils.read_image("defective-weld.png")

    seed_points = [  # (row, column)
        [254, 138],  # Seed point 1
        [253, 296],  # Seed point 2
        [233, 436],  # Seed point 3
        [232, 417],  # Seed point 4
    ]
    intensity_threshold = 50
    segmented_image = region_growing(im, seed_points, intensity_threshold)

    assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
        im.shape, segmented_image.shape)
    assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
        segmented_image.dtype)

    segmented_image = utils.to_uint8(segmented_image)
    utils.save_im("defective-weld-segmented.png", segmented_image)
