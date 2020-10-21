import pathlib
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

output_dir_images = pathlib.Path("image_solutions")
output_dir_images.mkdir(exist_ok=True)

# Load the dataset and print some stats
batch_size = 64

# Use CrossEntropyLoss for multi-class classification
loss_function = torch.nn.CrossEntropyLoss()

# Hyperparameters
learning_rate = .0192
num_epochs = 5

print("Using CUDA:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())


def task_abd():
    print("Task A:")
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

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
            nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
            nn.Linear(28*28*1, 10)  # 28*28 input features, 10 outputs
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
    # 10 since mnist has 10 different classes
    expected_shape = (batch_size, 10)
    assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"

    # Define optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)

    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )
    train_loss_dict_non_normalized, test_loss_dict_non_normalized = trainer.train(
        num_epochs)

    final_loss, final_acc = utils.compute_loss_and_accuracy(
        dataloader_test, model, loss_function)
    print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

    # Normalize from here on
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])

    # We reset the manual seed to 0, such that the model parameters are initialized with the same random number generator.
    torch.random.manual_seed(0)
    np.random.seed(0)

    dataloader_train, dataloader_test = dataloaders.load_dataset(
        batch_size, image_transform)

    model = create_model()

    # Redefine optimizer, as we have a new model.
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)
    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )
    train_loss_dict_normalized, test_loss_dict_normalized = trainer.train(
        num_epochs)

    # Plot loss
    utils.plot_loss(train_loss_dict_non_normalized,
                    label="Train Loss - Not normalized")
    utils.plot_loss(test_loss_dict_non_normalized,
                    label="Test Loss - Not normalized")
    utils.plot_loss(train_loss_dict_normalized,
                    label="Train Loss - Normalized")
    utils.plot_loss(test_loss_dict_normalized,
                    label="Test Loss - Normalized")
    # Limit the y-axis of the plot (The range should not be increased!)
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("Global Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig("image_solutions/task_4a.png")
    plt.clf()

    final_loss, final_acc = utils.compute_loss_and_accuracy(
        dataloader_test, model, loss_function)
    print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")

    ####################
    # Task B
    ####################
    print("Task B:")

    weight_image_array = np.zeros(shape=(28, 28))
    weight_tensors = list(model.children())[1].weight.cpu().data

    # 10 tensors since we have 0-9 classes
    for tensor_index, tensor in enumerate(weight_tensors):
        # Each tensor has length 28x28
        for index, value in enumerate(tensor):
            weight_image_array[index // 28, index % 28] = value

        utils.save_im(output_dir_images.joinpath("weights{}.jpg".format(
            tensor_index)), weight_image_array)

    ####################
    # Task D
    ####################
    print("Task D:")

    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])

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
            nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
            nn.Linear(28*28, 64),  # 28*28 input features, 64 outputs
            nn.ReLU(),  # ReLU as activation funciton for the layer above
            nn.Linear(64, 10),  # 64 inputs, 10 outputs
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
    # 10 since mnist has 10 different classes
    expected_shape = (batch_size, 10)
    assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"

    # Define optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=learning_rate)

    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )

    train_loss_dict_hidden, test_loss_dict_hidden = trainer.train(num_epochs)

    # Plot loss
    utils.plot_loss(train_loss_dict_normalized,
                    label="Train Loss - Normalized")
    utils.plot_loss(test_loss_dict_normalized,
                    label="Test Loss - Normalized")
    utils.plot_loss(train_loss_dict_hidden,
                    label="Train Loss - One hidden layer")
    utils.plot_loss(test_loss_dict_hidden,
                    label="Test Loss - One hidden layer")
    # Limit the y-axis of the plot (The range should not be increased!)
    plt.ylim([0, 1])
    plt.legend()
    plt.xlabel("Global Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig("image_solutions/task_4d.png")
    # plt.show()
    plt.clf()

    final_loss, final_acc = utils.compute_loss_and_accuracy(
        dataloader_test, model, loss_function)
    print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")


def task_c():
    print("Task C:")
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=0.5, std=0.5)
    ])

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
            nn.Flatten(),  # Flattens the image from shape (batch_size, C, Height, width) to (batch_size, C*height*width)
            nn.Linear(28*28*1, 10)  # 28*28 input features, 10 outputs
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
    # 10 since mnist has 10 different classes
    expected_shape = (batch_size, 10)
    assert output.shape == expected_shape,    f"Expected shape: {expected_shape}, but got: {output.shape}"

    new_learning_rate = 1.0

    # Define optimizer (Stochastic Gradient Descent)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=new_learning_rate)

    trainer = Trainer(
        model=model,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        batch_size=batch_size,
        loss_function=loss_function,
        optimizer=optimizer
    )
    train_loss_dict, test_loss_dict = trainer.train(num_epochs)

    # Plot loss
    utils.plot_loss(train_loss_dict, label="Train Loss - Learning rate = 1.0")
    utils.plot_loss(test_loss_dict, label="Test Loss - Learning rate = 1.0")
    # Limit the y-axis of the plot (The range should not be increased!)
    plt.ylim([0, 20])
    plt.legend()
    plt.xlabel("Global Training Step")
    plt.ylabel("Cross Entropy Loss")
    plt.savefig("image_solutions/task_4c.png")

    # plt.show()
    plt.clf()

    final_loss, final_acc = utils.compute_loss_and_accuracy(
        dataloader_test, model, loss_function)
    print(f"Final Test loss: {final_loss}. Final Test accuracy: {final_acc}")


task_abd()
task_c()
