import pathlib

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

def calc_gaussian(xx, yy, point, sigma):
    return np.exp(-((xx-point[1])**2 + (yy-point[0])**2) / sigma)

# Generate a "gaussian target" heatmap that has the following properties:
#   - Is nonzero everywhere
#   - Is composed of several gaussians of decreasing spread, to incentivize precision in the model
#   - Is a probability distribution (sum of all values is 1)
def generate_gaussian_target(point, imsize=(512, 512)):
    x = np.arange(0, imsize[1])
    y = np.arange(0, imsize[0])
    xx, yy = np.meshgrid(x, y)

    # The target is a sum of gaussians of varying sizes.
    heatmap = np.zeros((imsize[0], imsize[1]))
    # for i in range(6):
    #     heatmap += 1/(i+1) * calc_gaussian(xx, yy, point, 8**i)
    for i in range(6):
        heatmap += 1/(i+1) * calc_gaussian(xx, yy, point, 5**(i+1))

    # Turn into a probability distribution
    heatmap = torch.tensor(heatmap)
    heatmap = torch.softmax(heatmap.view(-1), dim=0).view(512, 512)
    heatmap = heatmap.numpy()

    return heatmap

""" --------------- Dataset --------------- """
class MaskData(Dataset):
    def __init__(self, training_dat = True, corners_only=False, randomization=True):
        self.randomization = randomization
        self.final_channels = 8
        if corners_only:
            self.final_channels = 4

        # Determine proper path given if this is the training or test set
        fix = "Train" if training_dat else "Test"

        df = pd.read_csv(f'Spatial_Training/Data/PointData{fix}.csv').to_numpy()

        self.len = df.shape[0]

        self.im_dirs = np.empty(self.len, dtype=object)
        self.corners = np.zeros((self.len, self.final_channels, 2))

        for i in range(self.len):
            file_dir = str(df[i,0])
            corner_info = np.int32(df[i,1:])
            self.im_dirs[i] = 'Spatial_Training/' + file_dir

            coef = 2
            if corners_only:
                coef = 4

            # Grab the points of interest
            for k in range(self.final_channels):
                self.corners[i, k, 0] = corner_info[k*coef]
                self.corners[i, k, 1] = corner_info[k*coef+1]
            

    def __getitem__(self, item):
        # Set up an affine transformation for if randomization shall occur
        aff = None
        if self.randomization:
            center = (512 // 2, 512 // 2)
            angle = np.random.uniform(-20, 20)
            scale = np.random.uniform(0.5, 1.0)

            aff = cv2.getRotationMatrix2D(center, angle, scale)

            # Random chance to just get the identity matrix (nothing changes)
            if np.random.rand() < 0.3:
                aff = np.array([
                    [1, 0, 0],
                    [0, 1, 0]
                ], dtype=np.float32)

        im = cv2.imread(self.im_dirs[item])[:,:,::-1]

        # Save the image's original size for later rescaling of points
        original_size = im.shape

        # Shrink the image to be compatible with the model
        im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)

        # Make the entries be between 0 and 1
        im = im.astype(np.float32)
        im /= 255.0

        if self.randomization:
            # Transform the image if performing randomization
            im = cv2.warpAffine(im, aff, (im.shape[1], im.shape[0])) #, borderMode=cv2.BORDER_REPLICATE)

        # Load in the point of interest masks
        mini_size = 512
        corners = np.zeros((self.final_channels, mini_size, mini_size))
        for i in range(self.final_channels):
            proper_corners = self.corners[item, i, :].copy()

            # Transform the corners to be in (512, 512) space
            proper_corners[0] = proper_corners[0] * (mini_size / original_size[0])
            proper_corners[1] = proper_corners[1] * (mini_size / original_size[1])

            # Transform the corners if an affine transformation is being applied
            if self.randomization:
                # An extra entry must be appended for the matrix multiplication to work
                pt = np.array([proper_corners[1], proper_corners[0], 1])
                pt = aff @ pt
                proper_corners[0] = pt[1]
                proper_corners[1] = pt[0]

            # Ensure corners are valid pixels
            proper_corners = np.int32(proper_corners)

            # Generate the gaussian heatmap
            corners[i] = generate_gaussian_target(proper_corners)
        corners = corners.astype(np.float32)

        return torch.tensor(im), torch.tensor(corners)

    def __len__(self):
        return self.len



""" --------------- Model --------------- """
class CornerNet(nn.Module):
    def __init__(self, corners_only=False):
        super(CornerNet, self).__init__()

        self.final_channels = 8
        if corners_only:
            self.final_channels = 4

        # Initial input size: 3 x 512 x 512
        # Note - Convolutional layers have padding since the original image dimensions are
        # powers of 2, which are very convenient for pooling.

        # Perform initial normalization
        self.norm = nn.BatchNorm2d(3)

        # Block 1 - Perform convolution followed by ReLU, twice
        # A dropout 2d is applied to reduce overfitting
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(3,3), padding='same'), # 32 x 512 x 512
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding='same'), # 32 x 512 x 512
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # Part of the result is copied for a later step, while the rest gets max pooled.
        # Down result: 32 x 256 x 256

        # Block 2 - Perform another two convolution + ReLU sequences and double the channels.
        # Also performs batch normalization to limit overfitting.
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3), padding='same'), # 64 x 256 x 256
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding='same'), # 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # Yet again, part of the result is copied, and then the remaining gets pooled.
        # Down result: 64 x 128 x 128

        # Block 3 - Same as before, more convolutions, ReLU and doubled channels
        # A dropout 2d is applied to reduce overfitting.
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3), padding='same'), # 128 x 128 x 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding='same'), # 128 x 128 x 128
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # Yet again, part of the result is copied, and then the remaining gets pooled.
        # Down result: 128 x 64 x 64

        # Block 4 - Same as before, more convolutions, ReLU and doubled channels
        # Also performs batch normalization to limit overfitting.
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), padding='same'),  # 256 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding='same'),  # 256 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )

        # The copied results of block 4 will be the last of the copied images.
        # The result then travels to the "bottom" of the U

        # A max pool is applied before the bottom layer.
        # Down result: 256 x 32 x 32

        # Bottom Layer - Channels are doubled once again and two convolutions are applied.
        # Dropout layer is added to make sure each of the channels are being useful
        self.bottom = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=(3,3), padding='same'), # 512 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3,3), padding='same'), # 512 x 32 x 32
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # Now we travel back up the U.
        # First, we apply a transposed convolution to the bottom result to increase its size.
        # Side note - could a similar thing be accomplished with a 3x3 kernel?

        # Note that there is a reduction in feature size, due to the fact some features will be copied over
        # from previous blocks.
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=2) # 256 x 64 x 64

        # Here, our output is combined with the copied answer from block 4, resulting in double the channels. - 512 x 64 x 64
        # This is then reduced as the system is fed into two convolution + ReLU layers.
        # Also performs batch normalization to limit overfitting.
        self.upblock1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=(3,3), padding='same'), # 256 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3,3), padding='same'), # 256 x 64 x 64
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.1)
        )

        # Upsampling is performed again. We use a transpose convolution to double the image size.
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=2) # 128 x 128 x 128

        # This doubled result is then combined with the result copied over from block 3, doubling the channels. - 256 x 128 x 128
        # This high channel image is then convolved over. Dropout2d is applied to reduce overfitting.
        self.upblock2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3,3), padding='same'), # 128 x 128 x 128
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3,3), padding='same'), # 128 x 128 x 128
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # Another upsampling using a transpose convolution to double image size.
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=(2,2), stride=2) # 64 x 256 x 256

        # This doubled result is then combined with the result copied over from block 2, doubling the channels. - 128 x 256 x 256
        # This high channel image is then convolved over.
        # Also performs batch normalization to limit overfitting.
        self.upblock3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3), padding='same'), # 64 x 256 x 256
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), padding='same'), # 64 x 256 x 256
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )

        # A final upsampling returns the image back into 512 x 512
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=(2,2), stride=2) # 32 x 512 x 512

        # This doubled result is then combined with the result copied over from block 2, doubling the channels. - 64 x 512 x 512
        # This high channel image is then convolved over.
        # Since this is the last layer, a 1x1 convolution is applied to the image to determine classifications for each mask.
        self.upblock4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3), padding='same'), # 32 x 512 x 512
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3,3), padding='same'), # 32 x 512 x 512
            nn.ReLU(),
            # The final layer is a 1x1 convolution that compresses the high channel image into a
            # heatmap across a four-channel image.
            nn.Conv2d(32, self.final_channels, kernel_size=(1,1))
        )

        # Finally, do a softmax to make each channel a probability distribution
        # Since torch.softmax only works over a single channel, compress along final two axes
        self.softmax = lambda x: torch.softmax(x.view(x.size(0), x.size(1), -1), dim=2).view_as(x)

    def forward(self, x):
        # Begin by normalizing the data
        x = self.norm(x)

        # Initial data size: 3 x 512 x 512

        # Apply the first block and save its result for later
        x = self.block1(x) # 32 x 512 x 512
        block_1_res = x.clone()

        # Continue down the U - use max pool to reduce image size
        x = F.max_pool2d(x, kernel_size=(2,2)) # 32 x 256 x 256

        # Apply the second block and save its result for later
        x = self.block2(x) # 64 x 256 x 256
        block_2_res = x.clone()

        # Continue down the U - use max pool to reduce image size
        x = F.max_pool2d(x, kernel_size=(2,2)) # 64 x 128 x 128

        # Apply the third block and save its result for later
        x = self.block3(x) # 128 x 128 x 128
        block_3_res = x.clone()

        # Continue down the U - use max pool to reduce image size
        x = F.max_pool2d(x, kernel_size=(2,2)) # 128 x 64 x 64

        # Apply the fourth block and save its result for later
        x = self.block4(x) # 256 x 64 x 64
        block_4_res = x.clone()

        # Enter the bottom the U - use max pool to reduce image size a final time
        x = F.max_pool2d(x, kernel_size=(2,2)) # 256 x 32 x 32

        # Apply the bottom layer of the network - this one's result is not saved
        x = self.bottom(x) # 512 x 32 x 32

        # Upsample x using a transpose convolution, then combine the result with block 4's result
        x = self.up1(x) # 256 x 64 x 64
        x = torch.cat((x, block_4_res), dim=1) # 512 x 64 x 64

        # Apply the first upblock which reduces channel count via convolution.
        x = self.upblock1(x) # 256 x 64 x 64

        # Again, upsample x using a transpose convolution, then combine with block 3's result
        x = self.up2(x) # 128 x 128 x 128
        x = torch.cat((x, block_3_res), dim=1) # 256 x 128 x 128

        # Apply the second upblock which reduces channel count via convolution
        x = self.upblock2(x) # 128 x 128 x 128

        # Upsample x using a transpose convolution, then combine with block 2's result
        x = self.up3(x) # 64 x 256 x 256
        x = torch.cat((x, block_2_res), dim=1) # 128 x 256 x 256

        # Apply the third upblock which reduces channel count via convolution
        x = self.upblock3(x) # 64 x 256 x 256

        # Perform the final upsampling to return to original image size
        # Combine this with the results from the first block
        x = self.up4(x) # 32 x 512 x 512
        x = torch.cat((x, block_1_res), dim=1) # 64 x 512 x 512

        # Apply the final block which will transform x into our mheatmap
        x = self.upblock4(x)

        # Put the image through a softmax to ensure they are probability distributions.
        x = self.softmax(x)

        return x