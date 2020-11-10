import numpy as np
import skimage
import utils
import pathlib
import matplotlib.pyplot as plt


def otsu_thresholding(im: np.ndarray) -> int:
    """
        Otsu's thresholding algorithm that segments an image into 1 or 0 (True or False)
        The function takes in a grayscale image and outputs a boolean image

        args:
            im: np.ndarray of shape (H, W) in the range [0, 255] (dtype=np.uint8)
        return:
            (int) the computed thresholding value
    """
    assert im.dtype == np.uint8
    # START YOUR CODE HERE ### (You can change anything inside this block)
    # You can also define other helper functions
    # Compute normalized histogram

    INTENSITY_RANGE = 255  # Number of grayscale values [0-255]

    threshold = 128
    hist, bins = np.histogram(im, bins=INTENSITY_RANGE)

    # Normalize hist
    # New array so we don't mix up float and ints
    hist_normalized = np.empty(INTENSITY_RANGE, dtype=float)
    number_of_pixels = im.shape[0] * im.shape[1]
    for i in range(len(hist)):
        hist_normalized[i] = float(hist[i]) / float(number_of_pixels)

    # Cumulative sums
    cumulative_sums = np.cumsum(hist_normalized)

    cumulative_means = np.empty(INTENSITY_RANGE, dtype=float)
    cumulative_means[0] = 0
    # Cumulative means
    for i in range(1, len(hist_normalized)):
        cumulative_means[i] = cumulative_means[i - 1] + i * hist_normalized[i]

    # Global mean
    global_mean = 0
    for i in range(len(hist_normalized)):
        global_mean += i * hist_normalized[i]

    # Between-class variances
    between_class_variances = np.empty(INTENSITY_RANGE, dtype=float)
    between_class_variances = (global_mean * cumulative_sums -
                               cumulative_means)**2 / (cumulative_sums * (1 - cumulative_sums))

    # Find Otsu threshold from maximum values in the between-class variances
    max_indexes = []
    max_value = max(between_class_variances)
    for i in range(len(between_class_variances)):
        if(between_class_variances[i] == max_value):
            max_indexes.append(i)

    threshold = sum(max_indexes) / len(max_indexes)

    # plt.figure(figsize=(16, 5))
    # columns = 1
    # rows = 4
    # plt.subplot(columns, rows, 1)
    # plt.plot(hist_normalized)
    # plt.title("Hist normalized")
    # plt.subplot(columns, rows, 2)
    # plt.plot(cumulative_sums)
    # plt.title("Cumulative sums")
    # plt.subplot(columns, rows, 3)
    # plt.plot(cumulative_means)
    # plt.title("Cumulative means")
    # plt.subplot(columns, rows, 4)
    # plt.plot(between_class_variances)
    # plt.title("Between-class variance")
    # plt.savefig("image_processed/task2a-full{}.png".format(round(threshold, 0)))
    # plt.show()

    return threshold
    ### END YOUR CODE HERE ###


if __name__ == "__main__":
    # DO NOT CHANGE
    impaths_to_segment = [
        pathlib.Path("thumbprint.png"),
        pathlib.Path("polymercell.png")
    ]
    for impath in impaths_to_segment:
        im = utils.read_image(impath)
        threshold = otsu_thresholding(im)
        print("Found optimal threshold:", threshold)

        # Segment the image by threshold
        segmented_image = (im >= threshold)
        assert im.shape == segmented_image.shape, "Expected image shape ({}) to be same as thresholded image shape ({})".format(
            im.shape, segmented_image.shape)
        assert segmented_image.dtype == np.bool, "Expected thresholded image dtype to be np.bool. Was: {}".format(
            segmented_image.dtype)

        segmented_image = utils.to_uint8(segmented_image)

        save_path = "{}-segmented.png".format(impath.stem)
        utils.save_im(save_path, segmented_image)
