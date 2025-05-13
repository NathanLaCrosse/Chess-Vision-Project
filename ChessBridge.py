import numpy as np
import Adapter as a
import matplotlib.pyplot as plt
import cv2
import CornerNet
from ultralytics import YOLO

"""
Generate a FEN string representation of a chess board from the image's file path. The current version 
cannot distinguish between piece types - they will all be classified as pawns.

Params:
    - path: The path to the desired input image.
    - model: The YOLO model to use to generate bounding boxes. Auto-loads if None.
    - conf: The confidence level for the YOLO model, if it is being auto-created.
    
Returns:
    - fen: A fen string representation of the current board state.
"""
def generate_FEN_from_image(path, model=None, conf=0.5):
    # Read the image
    im = cv2.imread(path)[:,:,::-1]

    # Load up the YOLO model if not given
    if model is None:
        model = YOLO('Yolo_Training/yolo_chess/weights/best.pt')

    # Get the resulting bounding box data
    results = model.predict(source=im, conf=conf)[0]

    # We need to keep track of classifications and where they are
    names = results.names                   # dictionary of cl, label pairs
    classifications = results.boxes.cls
    bounding_boxes = results.boxes.xyxyn

    # Get bottom-middle of each bounding box
    ymaxes = bounding_boxes[:, 3]
    xmids = (bounding_boxes[:, 0] + bounding_boxes[:, 2]) / 2.0
    box_midpoints = np.concatenate((ymaxes.reshape(-1, 1), xmids.reshape(-1, 1)), axis=1)

    # Get labels for each bounding box
    cl = classifications.numpy().copy().astype(object)
    labels = []
    for i in range(cl.shape[0]):
        label = names[cl[i]]

        if label == 'chess_piece':
            # Default value for now
            labels.append('white_pawn')
    labels = np.array(labels)

    # The spatial reasoning model uses a resized image
    spatial_reasoning_im = a.read_sanitized_image(path)

    # Input into the FEN string generator
    return convert_bounding_boxes_to_FEN(box_midpoints, labels, spatial_reasoning_im)

"""
Converts bounding boxes given by a YOLO model and places them on the virtual board, generating
a FEN string in the process. To do this, the spatial reasoning model is loaded up, performed
on the target image, and its lines are used to determine the file and rank of a piece.

Params:
    - box_midpoints: An N X 2 array with rows in the form (row, col) of the bottom midpoint of the box
                     These midpoints have their location normalized - the values should be between
                     0 and 1.
    - labels: The labels corresponding to the piece type of each of the box_midpoints boxes.
    - im: The target image, stored as an 512 x 512 x 3 array. It must be resized before using this method.
    - white_to_move: True if white has the next move. True by default.
    
Returns:
    - fen: A FEN string representing the board's current state.
"""
def convert_bounding_boxes_to_FEN(box_midpoints, labels, im, white_to_move = True):
    spatial_reasoning = a.instantiate_unet(path="CornerNet.pt", corners_only=True)

    # Convert bounding boxes from 0..1 to pixel space
    # The spatial reasoning model converts everything to 512x512, so that will be our conversion factor
    box_midpoints = box_midpoints * 512

    # Create our virtual board to put the pieces on
    virt = [[" " for _ in range(8)] for _ in range(8)]

    # Generate the predicted chessboard layout to assign rows and columns to pieces
    pred = a.predict_results(spatial_reasoning, im)
    pois = a.determine_pois_via_convolution(pred)
    rim = a.interp_rim(pois)
    h_lines, v_lines = a.interp_chessboard_lines(rim)

    # For each box, figure out where it is on the virtual board
    for i in range(labels.shape[0]):
        row, col = a.determine_row_col_for_point(rim, h_lines, v_lines, (box_midpoints[i,0], box_midpoints[i,1]))

        virt[row][col] = str(labels[i])

    # Final step - based on the way we are setting up pieceStr we create the FEN string.
    fen = ""

    # Let's keep track of whether we found the white and black kings - important
    # for the FEN string to work
    found_white_king = False
    found_black_king = False

    for i in range(8):
        empty_counter = 0

        for j in range(8):
            val = virt[i][j]

            # If the space is empty, simply increment the empty counter
            if val == " ":
                empty_counter += 1
            # If the space has something, add the empty counter and then the piece itself
            else:
                if empty_counter != 0:
                    fen = fen + str(empty_counter)
                    empty_counter = 0

                set_val = " "
                match val:
                    # black pieces -> pawn through king
                    case "black_pawn":
                        set_val = "P"
                    case "black_bishop":
                        set_val = "B"
                    case "black_knight":
                        set_val = "N"
                    case "black_rook":
                        set_val = "R"
                    case "black_queen":
                        set_val = "Q"
                    case "black_king":
                        set_val = "K"
                        found_black_king = True
                    # white pieces -> pawn through king
                    case "white_pawn":
                        set_val = "p"
                    case "white_bishop":
                        set_val = "b"
                    case "white_knight":
                        set_val = "n"
                    case "white_rook":
                        set_val = "r"
                    case "white_queen":
                        set_val = "q"
                    case "white_king":
                        set_val = "k"
                        found_white_king = True

                fen = fen + set_val

        # Add any remaining empty spaces to the FEN row
        if empty_counter != 0:
            fen = fen + str(empty_counter)

        # Add an end line character (if not on the final line)
        if i < 7:
            fen = fen + "/"

    # Now the FEN string contains where all of the pieces are on the board.
    # For the other metadata, we will be setting it blank since we don't know about
    # what moves have been made over the course of the game.

    # Who has the next move? Specified by the white_to_move parameter
    if white_to_move:
        fen = fen + " w"
    else:
        fen = fen + " b"

    # We assume that both sides can be allowed to castle in this position
    fen = fen + " -"

    # We assume there is no en passant available
    fen = fen + " -"

    # We assume halfmove counters are zero - we would need previous moves to determine this
    # Final segment - we assume we're on the first turn as we have no way of knowing how
    # far along the game has been.
    fen = fen + " 0 1"

    # Double check to see if both kings were included. If not, our FEN string will not work
    if not found_white_king or not found_black_king:
        print("WARNING - NOT ALL KINGS FOUND IN FEN STRING CONSTRUCTION")

    return fen