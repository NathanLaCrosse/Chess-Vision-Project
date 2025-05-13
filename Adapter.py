import numpy as np
import cv2
import torch
from Spatial_Training.CornerNet import CornerNet
from scipy.signal import fftconvolve

"""
Return a properly loaded in image. Note that cv2.imread reads in as BGR, which causes problems.
Also normalizes the image to have values between 0 and 1 and resizes it to be 512 by 512.
"""
def read_sanitized_image(path):
    im = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (512, 512), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32)
    return im / 255.0

"""Return a loaded heatmap UNet already in evaluation mode, ready to be used."""
def instantiate_unet(path, corners_only=False):
    # load the network dictionary
    net_dict = torch.load(path, map_location=torch.device('cpu'))

    net = CornerNet(corners_only=corners_only)
    net.load_state_dict(net_dict)

    return net.eval()

"""
Returns the predicted heatmap masks evaluated by the neural network. 

Params:
    - net: The neural network to evaluate the image.
    - im: The input image, stored as a 3 channel numpy array.

Returns:
    - heatmaps: The resulting heatmaps, stored as a torch tensor. Has dimension 4 x N x M
"""
def predict_results(net, im):
    with torch.no_grad():
        predicted = net(torch.tensor(im).view(1, 3, im.shape[0], im.shape[1]))
        return predicted.view(-1, im.shape[0], im.shape[1])

"""
Returns the maximum indices of the inputted heatmaps.

Params:
    - heatmaps: The heatmaps to process. Stored as N x W x H arrays.
    
Returns:
    - indices: An N X 2 array storing the row and column of the points of interest.
"""
def determine_pois(heatmaps):
    indices = np.zeros((heatmaps.shape[0], 2))

    for i in range(indices.shape[0]):
        dex = np.argmax(heatmaps[i])

        indices[i,0] = dex // heatmaps.shape[1]
        indices[i,1] = dex % heatmaps.shape[2]

    return np.int64(indices)

"""
Applies a convolution looking for blobs to the inputted heatmaps and returns the indices
of the maximum points in these convolved heatmaps. Increases general accuracy of the point prediction
model.

Params:
    - heatmaps: The heatmaps to process. Stored as N x W x H arrays.

Returns:
    - indices: An N X 2 array storing the row and column of the points of interest.
"""
def determine_pois_via_convolution(heatmaps):
    indices = np.zeros((heatmaps.shape[0], 2))

    # Generate a blob-shaped kernel to apply to the heatmaps
    ker_size = 41
    ker = np.zeros((ker_size, ker_size))
    radius = (ker_size + 1) // 2
    for i, j in np.ndindex(ker.shape):
        val = (i - radius - 1) ** 2 + (j - radius - 1) ** 2

        if val < radius ** 2:
            ker[i, j] = radius ** 2 - val

    for i in range(indices.shape[0]):
        # Apply the convolution and crop the added pixels.
        res = fftconvolve(heatmaps[i], ker)
        res = res[radius-1:-(radius-1), radius-1:-(radius-1)]

        dex = np.argmax(res)

        indices[i, 0] = dex // heatmaps.shape[1]
        indices[i, 1] = dex % heatmaps.shape[2]

    return np.int64(indices)

"""A helper method to calculate the midpoints in between the corners of the chessboard."""
def interp_points_from_corners(corners):
    inbetween = []
    for i in range(4):
        ni = (i + 1) % 4

        inbetween.append(corners[i])

        interp = (1/2)*(corners[ni] - corners[i]) + corners[i]
        inbetween.append(interp)

    return interp_rim_from_eight(np.array(inbetween))

"""
Generates the 32 points around the rim of the chessboard where tiles intersect.

Params:
    - pois: A 8 x 2 array of the four corners of the chessboard and the points between them.

Returns:
    - rim: A 32 x 2 array of the 32 key points around the rim of the chessboard.
"""
def interp_rim_from_eight(pois):
    # Determine the change between each point of interest
    deltas = [0] * 8
    for i in range(8):
        ni = (i + 1) % 8

        deltas[i] = [pois[ni, 0] - pois[i, 0], pois[ni, 1] - pois[i, 1]]
    deltas = np.array(deltas)

    rim = []
    # Interpolate across each point of interest to get all the points around the rim of the board
    for i in range(8):
        start = pois[i,:]
        delta = deltas[i,:]

        for k in range(4):
            rim.append((start + k/4*delta).astype(int))

    return np.array(rim)

"""
Generate the 32 points around the outside of the chessboard which are either corners or the points
where two tiles intersect.

Params:
    - pois: Either a 4 X 2 or 8 X 2 array of points of interest.

Returns:
    - rim: A 32 x 2 array of the 32 key points around the rim of the chessboard.
"""
def interp_rim(pois):
    if pois.shape[0] == 4:
        return interp_points_from_corners(pois)
    else:
        return interp_rim_from_eight(pois)

"""
Generate parametric curve descriptions of the lines in the chessboard whose rim is interpolated 
by rim.

Params:
    - rim: The 32 points around the rim of the chessboard where tiles intersect. 32 x 2 array.

Returns:
    - h_lines, v_lines: Parametric equations for the horizontal and vertical lines, stored
                        in the form (delta_row, row, delta_col, col). When t=1, the line
                        travels across the entire board.
"""
def interp_chessboard_lines(rim):
    # Group the points based on whether they are part of the top/bottom and/or left/right sides
    top_row = rim[0:9]
    bottom_row = rim[16:25]
    right_side = rim[8:17]
    left_side = np.zeros((9, 2))
    left_side[:8, :] = rim[24:32, :]
    left_side[8, :] = rim[0, :]

    # Generate parametric equations for both the vertical and horizontal lines.
    horizontal_lines = []
    for i in range(top_row.shape[0]):
        # Compute change in row and column
        delta = np.array([left_side[8 - i, 0] - right_side[i, 0], left_side[8 - i, 1] - right_side[i, 1]])

        start = left_side[i]

        horizontal_lines.append([delta[0], start[0], delta[1], start[1]])
    horizontal_lines = np.array(horizontal_lines)[1:-1] # Contains two extra lines that what we need (border edge)

    vertical_lines = []
    for i in range(top_row.shape[0]):
        # Compute change in row and column
        delta = np.array([bottom_row[8 - i, 0] - top_row[i, 0], bottom_row[8 - i, 1] - top_row[i, 1]])

        start = top_row[i]

        vertical_lines.append([delta[0], start[0], delta[1], start[1]])
    vertical_lines = np.array(vertical_lines)[1:-1] # Contains two extra lines that what we need (border edge)

    return horizontal_lines, vertical_lines

"""
Count the intersections between segment and the line for 0 < t < 1. Segment and lines are stored
as parametric curves across the image in question.

Curve format: (delta_row, starting_row, delta_col, starting_col)
"""
def count_intersections(segment, lines):
    intersections = 0
    # For each line, determine intersection time
    for i in range(lines.shape[0]):
        # Create an array and vector representing the system of equations to solve
        mat = np.array([
            [segment[0], lines[i,0]],
            [segment[2], lines[i,2]]
        ])
        vect = np.array([lines[i,1]-segment[1], lines[i,3]-segment[3]])

        intersect_times = np.linalg.solve(mat, vect)

        # Intersection time for main segment must be positive, less than one for the intersection to count
        if 0 < intersect_times[0] < 1:
            intersections += 1

    return intersections

"""
Determine the row and column of a point on the virtual chessboard.
Parametric curve format: (delta_row, starting_row, delta_col, starting_col)

Params:
    - rim: A 32 x 2 array consisting of row and column pairs for each "rim point" on the outside of
            the chessboard - where either tiles intersect or corners.
    - h_lines: A 7 X 4 array describing the parametric curves for the horizontal lines on the chessboard.
    - v_lines: A 7 x 4 array describing the parametric curves for the vertical lines on the chessboard.
    - point: A row and column pair that we want to determine the row and column of.

Returns:
    - row: The point's row on the virtual chessboard.
    - col: The point's column on the virtual chessboard.
"""
def determine_row_col_for_point(rim, h_lines, v_lines, point):
    # Our reference point will be in the top left corner
    segment = np.array([point[0] - rim[0, 0], rim[0, 0], point[1] - rim[0, 1], rim[0, 1]])

    # Count intersections with horizontal lines -> rows
    row = count_intersections(segment, h_lines)

    # Count intersections with vertical lines -> cols
    col = count_intersections(segment, v_lines)

    return row, col