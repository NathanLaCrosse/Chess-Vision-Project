import matplotlib.pyplot as plt
import Adapter as a
import Visualizer as v
from Spatial_Training.CornerNet import MaskData

# Load up the neural network and the dataset
corners_only = True
corners = 4 if corners_only else 8
net = a.instantiate_unet(path="CornerNet.pt", corners_only=corners_only)
dat = MaskData(training_dat=False, randomization=False, corners_only=corners_only)

# Check the virtual chessboard for each of the test images.
for i in range(len(dat)):
    im, mask = dat[i]
    im = im.numpy()

    pred = a.predict_results(net, im)
    pois = a.determine_pois_via_convolution(pred)
    rim = a.interp_rim(pois)

    plt.imshow(v.draw_chessboard(im, rim))
    plt.show()