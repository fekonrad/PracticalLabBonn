import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output
from time import perf_counter
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import random_split

# import libraries for data generation and U-Net implementation
from DataGeneration import generate_shapes
from DataGeneration import RandomShapes
from UNet import UNet
from UNet import EarlyStopping


""" Specify CPU or GPU """
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

""" Creating Dataset + DataLoaders """
num_images_train = 80
num_images_val = 20
num_images_test = 10

height = 128
width = 128
batch_size = 1

dataset_train = RandomShapes(num_images=num_images_train, height=height, width=width)
dataset_val = RandomShapes(num_images=num_images_val, height=height, width=width)
dataset_test = RandomShapes(num_images=num_images_test, height=height, width=width)

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, pin_memory=True)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True)

""" Hyperparameters for Training: """
lr = 0.001
momentum = 0.9
epochs = 100          # if early-stopping is used feel free to make 'epochs' very large.

"""number of channels to determine model size"""
num_channels = 16

model = UNet(num_channels=num_channels)
model.to(device)
loss_fn = nn.BCELoss()
# optimizer = optim.SGD(params=model.parameters(), lr=lr, momentum=momentum)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
early_stopper = EarlyStopping(patience=2, min_delta=0)

print(f'Number of trainable parameters: {model.get_size()}')

train_err = []
val_err = []
start = perf_counter()
for k in range(epochs):
    current_train_error = 0
    for img, mask in dataloader_train:
        img, mask = img.to(device), mask.to(device)
        y_pred = model(img)
        loss = loss_fn(y_pred, mask)
        current_train_error += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_err.append(current_train_error/len(dataloader_train))

    # calculate validation loss ...
    with torch.no_grad():
        current_val_error = 0
        for img, mask in dataloader_val:
            img, mask = img.to(device), mask.to(device)
            y_pred = model(img)
            val_loss = loss_fn(y_pred, mask)
            current_val_error += val_loss.item()
        current_val_error = current_val_error/len(dataloader_val)
        val_err.append(current_val_error)
    # Plot learning curves
    clear_output(wait=True)
    plt.plot(train_err, c='r')
    plt.plot(val_err, c='b')
    plt.title("Train Loss (Red) and Validation Loss (Blue) during training")
    plt.draw()
    plt.pause(0.1)  # Adjust the duration as needed

    if early_stopper.early_stop(current_val_error):
        break

train_time = perf_counter() - start
print(f'Training time for {num_images_train} data points and  {k+1} epochs: {int(train_time)} seconds')

"""Inspect a few predictions on the test set """
for img, mask in dataloader_test:
    img, mask = img.to(device), mask.to(device)
    y_pred = model(img)
    plt.clf()  # Clear the previous plot
    plt.subplot(1, 2, 1)
    plt.imshow(img[0].detach().numpy().transpose(1,2,0).astype(int))
    plt.title('Input Image')

    plt.subplot(1, 2, 2)
    img2 = plt.imshow(y_pred[0][0].detach().numpy())
    plt.colorbar(img2)
    plt.title('Predicted Mask')
    plt.tight_layout()
    plt.show()






