import torch
import matplotlib.pyplot as plt
import Visualizer as v

from CornerNet import MaskData, CornerNet

corners = 4

# Load up the dataset and neural network to test.
dat = MaskData(training_dat=False, randomization=False)
net_dict = torch.load('../CornerNet.pt', map_location=torch.device('cpu'))
net = CornerNet(corners_only= corners == 4)
net.load_state_dict(net_dict)

net = net.eval()

with torch.no_grad():
    for i in range(len(dat)):
        # Load an image and mask
        im, mask = dat[i]

        if corners == 4:
            mask = mask[::2,:,:]

        # Generate the network's mask prediction
        predicted_result = net(im.view(1,3,512,512))

        predicted_result = predicted_result.view(corners, 512, 512)

        # Plot the results
        v.visualize_results_with_traindat(im, mask, predicted_result)

        plt.show()