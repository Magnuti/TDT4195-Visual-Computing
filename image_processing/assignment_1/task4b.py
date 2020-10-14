import pathlib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from utils import save_im
import dataloaders
import torchvision
from trainer import Trainer
torch.random.manual_seed(0)
np.random.seed(0)

model = torch.load("models/model.pt")

output_dir = pathlib.Path("image_solutions")
output_dir.mkdir(exist_ok=True)

weight_image_array = np.zeros(shape=(28, 28))
weight_tensors = list(model.children())[1].weight.cpu().data


# 10 tensors since we have 0-9 classes
for tensor_index, tensor in enumerate(weight_tensors):
    # Each tensor has length 28x28
    for index, value in enumerate(tensor):
        weight_image_array[index // 28, index % 28] = value

    save_im(output_dir.joinpath("weights{}.jpg".format(
        tensor_index)), weight_image_array)
