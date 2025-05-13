import numpy as np
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm
from skimage import io, transform
from torchvision import transforms, utils

from CornerNet import MaskData, CornerNet
import Visualizer as v
import Adapter as a

""" --------------- Training --------------- """
def train_nn(epochs=5, batch_size=16, lr=0.001, epochs_per_save=20, file_prefix='PT Archive/post_trained_variety', loaded_net = None, corners=4):
    # Set up the data loader
    dat = MaskData(corners_only= corners == 4)
    dat_loader = DataLoader(dat, batch_size=batch_size, drop_last=True, shuffle=True)

    # Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create neural network and print parameter count
    net = loaded_net
    if loaded_net == None:
        net = CornerNet(corners_only= corners == 4)
    net = net.to(device)
    print(f"Total parameters: {sum(param.numel() for param in net.parameters())}")

    # Initialize our loss function - KLDivLoss
    # This takes the logarithm of our probability distribution, useful since the values
    # are so small. This loss function compares probability distribution functions, which
    # are our heatmaps for point prediction.
    loss_function = nn.KLDivLoss(reduction='batchmean')

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        for _, data in enumerate(tqdm.tqdm(dat_loader)):
            # Load in the data + bring it to the desired device
            x, y = data

            x = x.view(-1, 3, 512, 512)
            x = x.to(device)
            y = y.to(device)

            # Reset the optimizer
            optimizer.zero_grad()

            # Generate an output and determine loss
            output = net(x)
            loss = loss_function(torch.log(output), y)

            # Perform an optimization step
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Running loss for epoch {epoch + 1}: {running_loss:.4f}")
        running_loss = 0.0

        if (epoch + 1) // epochs_per_save and (epoch + 1) % epochs_per_save == 0:
            torch.save(net.state_dict(), f'{file_prefix}_{(epoch + 1)}.pt')
    
    return net, dat

corners = 8

net = None

net, dat = train_nn(epochs=1, lr=0.001, batch_size=8, epochs_per_save=50, file_prefix='Test',
                    loaded_net=net, corners=corners)
print('Training Finished!')

net = net.to(torch.device("cpu"))

# Code for testing results of the neural network
im, mask = dat[0]
net = net.eval()

# View a result to see if the model learned
# This is an image from the training data - use UNetTesting.py (or ChessboardTests.py) to see
# results on testing data.
with torch.no_grad():
    predicted_result = net(im.view(1,3,512,512))

    predicted_result = predicted_result.view(corners, 512, 512).numpy()

    v.visualize_results_with_traindat(im, mask, predicted_result)

    plt.show()
