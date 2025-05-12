import numpy as np
import Adapter as a
import matplotlib.pyplot as plt
import cv2
import CornerNet

"""
Converts bounding boxes given by a YOLO model and places them on the virtual board, generating
a FEN string in the process. To do this, the spatial reasoning model is loaded up, performed
on the target image, and its lines are used to determine the file and rank of a piece.

Params:
    - box_midpoints: An N X 2 array with rows in the form (row, col) of the bottom midpoint of the box
                     These midpoints have their location normalized - the values should be between
                     0 and 1.
    - labels: The labels corresponding to the piece type of each of the box_midpoints boxes.
    - im: The target image, stored as an M X N X 3 array.
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

# Testing code.
dat = CornerNet.MaskData(training_dat=False, randomization=False, corners_only=True)
boundingboxes = np.array([
    [230, 71],
    [170, 360],
    [127, 390]
])
boundingboxes = boundingboxes / 512.0
labels = np.array(["black_king", "white_rook", "black_bishop"])

im, _ = dat[5]
im = im.numpy()

fen = convert_bounding_boxes_to_FEN(boundingboxes, labels, im, True)

print(fen)

plt.imshow(im)
plt.show()