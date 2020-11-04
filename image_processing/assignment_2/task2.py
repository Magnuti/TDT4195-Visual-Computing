import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import utils
import dataloaders
import torchvision
from trainer import Trainer
torch.random.manual_seed(0)
np.random.seed(0)


# Load the dataset and print some stats
batch_size = 64

image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5], [0.5])])

dataloader_train, dataloader_test = dataloaders.load_dataset(
    batch_size, image_transform)
example_images, _ = next(iter(dataloader_train))
print(f"The tensor containing the images has shape: {example_images.shape} (batch size, number of color channels, height, width)",
      f"The maximum value in the image is {example_images.max()}, minimum: {example_images.min()}", sep="\n\t")


def create_model():
    """
        Initializes the mode. Edit the code below if you would like to change the model.
    """
    model = nn.Sequential(
        # First convolution
        nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Second convolution
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Third convolution
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        # Neural network
        nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
        # nn.Linear(4194304, 64),
        nn.Linear(32*32*2, 64),

        nn.ReLU(),
        nn.Linear(64, 10)
        # No need to include softmax, as this is already combined in the loss function
    )
    # Transfer model to GPU memory if a GPU is available
    model = utils.to_cuda(model)
    return model


model = create_model()

# Test if the model is able to do a single forward pass
example_images = utils.to_cuda(example_images)
output = model(example_images)
print("Output shape:", output.shape)
expected_shape = (batch_size, 10)  # 10 since mnist has 10 different classes
assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"

print("Using CUDA:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())

# Hyperparameters
learning_rate = .02  # Task 2a
num_epochs = 5


# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Define optimizer (Stochastic Gradient Descent)
optimizer = torch.optim.SGD(model.parameters(),  # Task 2a
                            lr=learning_rate)


start = time.time()

trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict, test_loss_dict = trainer.train(num_epochs)

print("Training 2a took {} seconds".format(round(time.time() - start, 3)))

# Task 2b

learning_rate = 0.001  # Task 2b

optimizer = torch.optim.Adam(model.parameters(),  # Task 2b
                             lr=learning_rate)

start = time.time()

trainer = Trainer(
    model=model,
    dataloader_train=dataloader_train,
    dataloader_test=dataloader_test,
    batch_size=batch_size,
    loss_function=loss_function,
    optimizer=optimizer
)
train_loss_dict_b, test_loss_dict_b = trainer.train(num_epochs)

print("Training 2b took {} seconds".format(round(time.time() - start, 3)))


# We can now plot the training loss with our utility script

# Plot loss
utils.plot_loss(train_loss_dict, label="Train Loss a")
utils.plot_loss(test_loss_dict, label="Test Loss a")
utils.plot_loss(train_loss_dict_b, label="Train Loss b")
utils.plot_loss(test_loss_dict_b, label="Test Loss b")
# Limit the y-axis of the plot (The range should not be increased!)
# plt.ylim([0, .5]) # Task 2a
plt.ylim([0, 1])  # Task 2b
plt.legend()
plt.xlabel("Global Training Step")
plt.ylabel("Cross Entropy Loss")
# plt.savefig(utils.image_output_dir.joinpath("task2a_plot.png")) # Task 2a
plt.savefig(utils.image_output_dir.joinpath("task2b_plot.png"))  # Task 2b
plt.show()

final_loss, final_acc = utils.compute_loss_and_accuracy(
    dataloader_test, model, loss_function)
print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")
