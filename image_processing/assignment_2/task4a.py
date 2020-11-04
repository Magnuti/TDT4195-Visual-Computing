import matplotlib.pyplot as plt
import numpy as np
import skimage
import utils


def convolve_im(im: np.array,
                fft_kernel: np.array,
                verbose=True):
    """ Convolves the image (im) with the frequency kernel (fft_kernel),
        and returns the resulting image.

        "verbose" can be used for turning on/off visualization
        convolution

    Args:
        im: np.array of shape [H, W]
        fft_kernel: np.array of shape [H, W] 
        verbose: bool
    Returns:
        im: np.array of shape [H, W]
    """
    # START YOUR CODE HERE ### (You can change anything inside this block)
    dft_image = np.fft.fft2(im)

    dft_image_filtered = dft_image * fft_kernel

    def to_real(dft_img):
        return np.sqrt(dft_image.real**2 + dft_image.imag**2)

    conv_result = np.fft.ifft2(dft_image_filtered).real
    if verbose:
        dft_image = np.fft.fftshift(dft_image)
        dft_image = to_real(dft_image)
        dft_image = np.log(dft_image + 1)

        fft_kernel = np.fft.fftshift(fft_kernel)

        dft_image_filtered = np.fft.fftshift(dft_image_filtered)
        dft_image_filtered = to_real(dft_image_filtered)
        dft_image_filtered = np.log(dft_image_filtered + 1)

        # Use plt.subplot to place two or more images beside eachother
        plt.figure(figsize=(16, 6))

        # plt.subplot(num_rows, num_cols, position (1-indexed))
        plt.subplot(1, 5, 1)
        plt.imshow(im, cmap="gray")

        # Visualize FFT
        plt.subplot(1, 5, 2)
        plt.imshow(dft_image)

        # Visualize FFT kernel
        plt.subplot(1, 5, 3)
        plt.imshow(fft_kernel)

        # Visualize filtered FFT image
        plt.subplot(1, 5, 4)
        plt.imshow(np.fft.fftshift(dft_image_filtered))

        # Visualize filtered spatial image
        plt.subplot(1, 5, 5)
        plt.imshow(conv_result, cmap="gray")

        first_value = fft_kernel[0][0]
        if(first_value == 1.0):
            plt.savefig(utils.image_output_dir.joinpath(
                "task4a_high_pass.png"))
        elif(first_value == 0.0):
            plt.savefig(utils.image_output_dir.joinpath(
                "task4a_low_pass.png"))
        else:
            plt.savefig(utils.image_output_dir.joinpath("task4a_unknown.png"))

    ### END YOUR CODE HERE ###
    return conv_result


if __name__ == "__main__":
    verbose = True
    # Changing this code should not be needed
    im = skimage.data.camera()
    im = utils.uint8_to_float(im)
    # DO NOT CHANGE
    frequency_kernel_low_pass = utils.create_low_pass_frequency_kernel(
        im, radius=50)
    image_low_pass = convolve_im(im, frequency_kernel_low_pass,
                                 verbose=verbose)
    # DO NOT CHANGE
    frequency_kernel_high_pass = utils.create_high_pass_frequency_kernel(
        im, radius=50)
    image_high_pass = convolve_im(im, frequency_kernel_high_pass,
                                  verbose=verbose)

    if verbose:
        plt.show()
    utils.save_im("camera_low_pass.png", image_low_pass)
    utils.save_im("camera_high_pass.png", image_high_pass)
