import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import Adapter as a
from ultralytics import YOLO
import ChessBridge as bridge

"""
Visualize the output masks of the neural network in regards to a certain image.

Params:
    - im: The input image, stored as a H X W X 3 numpy array.
    - results: The heatmaps to display, stored as a 4 x H x W array.

Returns:
    - fig, ax: fig and ax corresponding to the subplots used by the visualizer.
"""
def visualize_training_results(im, results):
    fig, ax = plt.subplots(2, results.shape[0])
    ax[0, 0].imshow(im)
    for i in range(4):
        ax[1, i].imshow(results[i], cmap='hot')
        ax[1, i].set_title(f"Predicted Mask {i+1}")

    plt.tight_layout()

    return fig, ax

"""
Visualize the output masks of the neural network as well as the original labels of an image.

Params:
    - im: The input image in the shape (H, W, 3)
    - true_labels: The true heatmasks, stored as a N X H X W array.
    - pred_labels: The predicted heatmasks, stored as a N X H X W array.
"""
def visualize_results_with_traindat(im, true_labels, pred_labels):
    fig, ax = plt.subplots(3, true_labels.shape[0])

    ax[0,0].imshow(im)

    # Clear all plots not being used
    blanks = ax[0,1:]
    for blank in blanks:
        blank.axis('off')

    for i in range(true_labels.shape[0]):
        ax[1, i].imshow(true_labels[i], cmap='hot')
        ax[1, i].set_title(f"True Mask {i+1}")
        ax[1, i].axis('off')

        ax[2, i].imshow(pred_labels[i], cmap='hot')
        ax[2, i].set_title(f"Predicted Mask {i + 1}")
        ax[2, i].axis('off')

    plt.tight_layout()

    return fig, ax


"""
Return a copied image that has points drawn on it. Default color is green.

Params:
    - im: The input image.
    - points: The input points, stored as a N x 2 array.

Returns:
    - new_im: The drawn-on image.
"""
def draw_point(im, points, size=5):
    assert len(points.shape) == 2, "Points should be a N x 2 array."

    new_im = im.copy()

    for point in points:
        new_im[int(point[0])-size:int(point[0])+size+1, int(point[1])-size:int(point[1])+size+1, :] = np.array([0,1.0,0])

    return new_im


"""
Overlay the predicted virtual chessboard on top of the original image. The produced image
is returned

Params:
    - im: The original image that the prediction was based on, as a H X W X 3 array.
    - border_points: The 32 points of interest along the border of the chessboard. Includes
                     the corners as well as where tiles intersect.

Returns:
    - drawn_im: The original image but with the chessboard lines overlayed onto it
"""
def draw_chessboard(im, border_points):
    # Group the points based on whether they are part of the top/bottom and/or left/right sides
    top_row = border_points[0:9]
    bottom_row = border_points[16:25]
    right_side = border_points[8:17]
    left_side = np.zeros((9, 2))
    left_side[:8, :] = border_points[24:32, :]
    left_side[8, :] = border_points[0, :]

    pts_per_line = 1000

    # Calculate horizontal line points.
    horizontal_line_pts = []
    for i in range(top_row.shape[0]):
        delta = np.array([bottom_row[8 - i, 0] - top_row[i, 0], bottom_row[8 - i, 1] - top_row[i, 1]])

        start = top_row[i]

        for k in range(pts_per_line):
            horizontal_line_pts.append(start + k / pts_per_line * delta)
    horizontal_line_pts = np.array(horizontal_line_pts)

    # Calculate vertical line points
    vertical_line_pts = []
    for i in range(left_side.shape[0]):
        delta = np.array([right_side[left_side.shape[0] - 1 - i, 0] - left_side[i, 0],
                          right_side[left_side.shape[0] - 1 - i, 1] - left_side[i, 1]])

        start = left_side[i]

        for k in range(pts_per_line):
            vertical_line_pts.append(start + k / pts_per_line * delta)
    vertical_line_pts = np.array(vertical_line_pts)

    # Combine all of the line points
    line_pts = np.concatenate((horizontal_line_pts, vertical_line_pts), axis=0)

    return draw_point(im, line_pts, size=2)

"""
Generate both YOLO object detection and virtual chessboard results for a given image, 
given as the file path. This result is plotted to matplotlib.

Params:
    - im_path: The path to the desired image file.
    - spatial_model: The spatial reasoning model. Will auto-load if None.
    - yolo_model: The YOLO classifier. Will auto-load if None.
    - conf: The confidence level for YOLO.

Returns:
    - None
"""
def visualize_final_results(im_path, spatial_model=None, yolo_model=None, conf=0.5):
    if spatial_model is None:
        spatial_model = a.instantiate_unet(path="CornerNet.pt", corners_only=True)
    if yolo_model is None:
        yolo_model = YOLO('best.pt')

    spatial_im = a.read_sanitized_image(im_path)
    yolo_im = cv2.imread(im_path)[:, :, ::-1]

    # Gather the spatial reasoning results
    pred = a.predict_results(spatial_model, spatial_im)
    pois = a.determine_pois_via_convolution(pred)
    rim = a.interp_rim(pois)
    lines_im = draw_chessboard(spatial_im, rim)

    # Generate the YOLO results
    results = yolo_model.predict(source=im_path, conf=conf)
    boxes_im = results[0].plot(img=yolo_im)

    # Plot everything
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(lines_im)
    ax[0].axis('off')
    ax[0].set_title("Virtual Chessboard")

    ax[1].imshow(boxes_im)
    ax[1].axis('off')
    ax[1].set_title("YOLO Objects")

    # Print the generated FEN string based on the results to std out
    fen = bridge.generate_FEN_from_image(im_path, yolo_model, conf=conf)
    print(f"\nFEN string for image: {fen}")

    plt.show()