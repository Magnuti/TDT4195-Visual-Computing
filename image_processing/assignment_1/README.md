# Image processing assignment 1
There are two parts to this assignment:
* Image processing (blurring, edge detection, convolutions etc.)
* Handwritten digit classification with neural networks on the MNIST dataset 

## Getting started
```
conda install tqdm
conda install -c conda-forge scikit-image
```
PyTorch with CUDA:
```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
If we want to use the CPU only (not CUDA) we can run this command instead of the above: `$ conda install pytorch torchvision cpuonly -c pytorch`

## Image processing results
### Original image
<img src="images/lake.jpg" alt="Original image" title="Original image" width="300"/>

### Greyscaled image
<img src="image_solutions/lake_greyscale.jpg" alt="Greyscaled image" title="Greyscaled image" width="300"/>

### Inversed greyscale image
<img src="image_solutions/lake_greyscale_inversed.jpg" alt="Inversed greyscaled image" title="Inversed greyscale image" width="300"/>

### Vertical Sobel kernel for vertical edge detection
<img src="image_solutions/im_sobel.jpg" alt="Original image" title="Original image" width="300"/>

### Blurred original image
<img src="image_solutions/im_smoothed.jpg" alt="Blurred image" title="Blurred image" width="300"/>

## Neural network results
### Normalized vs. original data
As we can see, the neural network performs better when the data is normalized (i.e when the image is normalized between the range [-1, 1]).
<img src="image_solutions/task_4a.png" alt="Normalized vs. original" title="Normalized vs. original" width="500"/>

### Weights for each class
The images below represent the inputs that are important for a certain class (i.e. the weights). As an example, let us inspect 0. The 0 class cares most about whether the pixels around the center is lit up and not the center itself.

<img src="image_solutions/weights0.jpg" alt="Weight 0" title="Weight 0" width="100"/>
<img src="image_solutions/weights1.jpg" alt="Weight 1" title="Weight 1" width="100"/>
<img src="image_solutions/weights2.jpg" alt="Weight 2" title="Weight 2" width="100"/>
<img src="image_solutions/weights3.jpg" alt="Weight 3" title="Weight 3" width="100"/>
<img src="image_solutions/weights4.jpg" alt="Weight 4" title="Weight 4" width="100"/>
<img src="image_solutions/weights5.jpg" alt="Weight 5" title="Weight 5" width="100"/>
<img src="image_solutions/weights6.jpg" alt="Weight 6" title="Weight 6" width="100"/>
<img src="image_solutions/weights7.jpg" alt="Weight 7" title="Weight 7" width="100"/>
<img src="image_solutions/weights8.jpg" alt="Weight 8" title="Weight 8" width="100"/>
<img src="image_solutions/weights9.jpg" alt="Weight 9" title="Weight 9" width="100"/>

### Learning rate of 1.0
Normally, the learning rate is 0.192, but here we set it to 1.0. What happens is that the model performs worse than with a learning rate of 0.192. This is because the learning rate is too high, so the gradient descent algorithm fails to converge.

<img src="image_solutions/task_4c.png" alt="Too high learning rate" title="Too high learning rate" width="500"/>

### One hidden layer
With one hidden layer, we can see that the model learns better. The loss does not stop at 0.3 but keeps sinking down to 0.2 as the step increases
<img src="image_solutions/task_4d.png" alt="One hidden layer" title="One hidden layer" width="500"/>


