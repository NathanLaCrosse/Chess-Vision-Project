# Chess Vision Project 
The goal of this project is to convert a raw image file of a chess position to a FEN string (common chess notation) that could then be entered into a traditional chess engine. The end use is to expedite the process of setting up a chess position in a chess engine, mainly for those who play on physical boards and want to examine their performance in a game. Unfortunately, due to extenuating circumstances in the team and the small amounts of available data, the full implementation of this project is not yet complete. The completed workflow can identify where pieces are on the board, but not what kind of piece it is. Below are a few example results of the final project, both the good and bad:

<img width="461" alt="Connections1" src="https://github.com/user-attachments/assets/4112f32e-bb48-4576-a7c5-38a123a7b2d9" />

<img width="454" alt="Connections2" src="https://github.com/user-attachments/assets/edd31294-1872-429d-b87c-04ad38140644" />

<img width="432" alt="actual miracle" src="https://github.com/user-attachments/assets/340d1e64-418c-4e14-8d08-ccebe84cf75b" />

### A Note on FEN Strings
The first portion of the FEN string - which is the only thing the workflow changes - encodes the current position of pieces on the board. It starts at the top left corner of the board and reads left to right. Numbers denote empty space, and the amount of empty spaces. "p" represents a black pawn, which is the current placeholder for a piece. Slash marks denote a change in row, from top to bottom.

To generate these results on your own, please reference the files Fen String Testing.py and ChessBridge.py, which showcase and implement the main methods used in the workflow. The rest of this readme file is dedicated to explaining the finer details in solving the challenges of object detection and placing the pieces on the "virtual board".

# Chess Vision Project - YOLO Object Detection
YOLO - "You Only Look Once" is an object detection model that treats object detection as a single regression problem, predicting bounding boxes and class probabilities directly from the full image in one pass. YOLO models are widely used in various applications like autonomous driving, surveillance, and robotics. For this case, we will be using it to find the chess pieces on the board. 

Starting out, we found this online Kaggle dataset, which convinently had the bounding boxes data for us already: https://www.kaggle.com/datasets/josephnelson/chess-piece-images-and-bounding-boxes.
<br>
<br>
For this training we also used YOLO model 11m, which is a good middle tier of thier newest and highest model. 

<br>
Once all of that was situated, the training began.



<br>
In this final version of the trained YOLO Object Detection, there is still room for improvement in the accuraccy; however, it does work! 

![yolo example](https://github.com/user-attachments/assets/44fbcf6f-9bb8-4084-be09-c312ad958835)


As you can see, YOLO is able to determine not only where the chess pieces are with a decent accuraccy, but also give it these very helpful bounding boxes. Once we get these bounding boxes, we can move to the next step. 


# Chess Vision Project - Spatial Reasoning
With YOLO object detection, we can reliably draw bounding boxes around each chess piece in the image. However, for these bounding boxes to be useful, we needed a way to 
connect a position in the image to a position on the chessboard. 

After some deliberation, I came up with the following algorithm. If we knew where the lines of the chessboard existed in the image, we could count how many lines we crossed 
on the way to a given location to determine the rank and file of a position. In the following image, if we interpret the top-left corner as being (0,0) and the bottom right 
as being (7,7), then by counting the intersections, (5 horizontal, 1 vertical) we can determine that the white pawn is at the position (5, 1) on the board.

![chess_alg_example](https://github.com/user-attachments/assets/784bc20d-4e3c-436a-bcfd-8969e820bd3b)

Now this creates a new problem - we need to know where the lines of the chessboard exist. Traditional straight line detecting algorithms fail with these images, due to 
noise being added in with the inclusion of chess pieces and background objects (and often the board outline is detected too). Furthermore, this program should support 
images taken from any angle, producing plenty of edge cases that elevate the problem's difficulty.

To solve these issues, we employ the use of a convolutional neural network with a UNet structure, depicted below.

<img width="344" alt="arcitecture" src="https://github.com/user-attachments/assets/7783293d-23f9-4f52-8abc-7227c65fee85" />

To explain how this will find the lines, let's start with the data collected for this model. The raw data can be found at the following Kaggle link: 
https://www.kaggle.com/datasets/nathanlacrosse/chess-vision-dataset/data. The data consists on images and pixel position data for where the corners of the chessboard are. 
The model uses this data to predict where the corners of the chessboard are. The final layer of the model outputs four probability distributions for where that given corner 
is, starting with the top-left corner and moving clockwise.

Using this, we can obtain an image with the four corners of the chessboard overlaid on it:

<img width="272" alt="four_corners" src="https://github.com/user-attachments/assets/e2964354-4c93-4ae9-97ae-d30d27701c90" />

Now, we take advantage of the fact that the grid of the chessboard is evenly spaced. Note that this isn't precisely true due to the perspective's effect on the image, but 
it gives an approximation that we can work with. Using this approximation, we interpolate around the rim of the chessboard to obtain points that we will draw lines through.

<img width="274" alt="interpolated_rim" src="https://github.com/user-attachments/assets/3168de46-509d-4d94-b607-1f2a728c4c64" />

Finally, we connect the points with lines to obtain our "virtual chessboard".

<img width="273" alt="spat_result_1" src="https://github.com/user-attachments/assets/eb401299-07c2-4d19-ba37-380e27a6b31d" />

While the result isn't perfect, if we take the bottom-middle of each bounding box around each chess piece, obtained from the YOLO model, we can create a close
representation of the current board state.

## Technical Details - Heatmaps
The raw point position data from the dataset cannot be fed into the model directly. Furthermore, since we want our result from the model to be a probability distribution of 
where it thinks the given point should be, we'll need to do some preprocessing so that the model can have some distributions to compare with.

To create the heatmaps, we create an image that has values determined by a gaussian function centered around the position data, like so:

<img width="200" alt="heatmap" src="https://github.com/user-attachments/assets/0bc5a8a7-2f9f-44b4-a749-161c63cacaad" />

To make sure the model can learn given these heatmaps, we want the image to be nonzero everywhere but increase and accelerate as we get closer to the true point. This 
presents a bit of a problem since the gaussian scales off so quickly that it becomes zero far away from the point (due to rounding error). So, the above image is not 
actually a single gaussian - it's a linear combination of several with different spreads. This ensures the properties we want from the image: it is nonzero everywhere and 
increases greatly as you approach the point. Finally, a softmax is applied to the image to ensure that it is truly a probability distribution.

As a final note, the model uses KLDivLoss as a loss function. This function is designed to work with probability distributions and essentially measures the similarity 
between two distributions.

## Technical Details - Line Intersections
In order to determine line intersections, the lines found in our virtual chessboard creation process are stored as parametric curves, which are stored as a four element
array (delta_row, starting_row, delta_col, starting_col). The top-left corner is chosen as our point of reference, the distance of which will help us find the position
of a piece. We create another line segment from the top-left corner to wherever out point in question is. This line segment is also stored as a parametric curve in the same
format as mentioned previously.

Solving for the intersections is simply solving a system of equations for where the reference line segment is equal to a chessboard line's line segment. In order for this 
intersection to be valid, the time of the intersection must be between 0 and 1, or else the two lines intersect outside of their defined region. This whole process is set 
up as a matrix equation which is solved efficiently with NumPy. 

## Technical Details - Determining Predicted Points
At first glance, the predicted point of the model should be where the probability distribution is the largest - as this is where the point is most likely to be. However,
our heatmaps outputted by the model end up being blobs and there is variance in the location of the greatest point in this blob. Take a look at the predicted heatmaps below:

<img width="450" alt="PredictedMasks" src="https://github.com/user-attachments/assets/80de05ed-11d8-4c33-bcc3-5228a9057253" />

We can actually improve the accuracy of the point prediction if we instead decide to look for these blobs instead of just the brightest point. To do this, the image is
convolved over with a kernel designed to look for them. Then the brightest point in the convolved image corresponds to the brightest blob. Below are two images showcasing 
the accuracy gained with this improvement:

<img width="458" alt="convolution" src="https://github.com/user-attachments/assets/53f4650c-1267-4475-aa98-33463ebad8bc" />

While there is only a slight improvement in point prediction, accurate prediction is imperative since the entire chessboard is interpolated from just four points. 

## Technical Details - Data Augmentation
When the neural network was trained, the model was quite overfit and could not function on any of the test images. To help train the model, random rotations and stretches 
are sometimes (~70% chance to) applied to the image. A randomizied image may look like the one below. Also, it is important that the image is not made unrecognizable - it 
must make sense that the first point from the csv file lines up with the top-left corner of the chessboard in the image (no horizontal/vertical flips).

<img width="275" alt="randomization_example" src="https://github.com/user-attachments/assets/0d9aad41-12c7-498a-92e4-785b4eae2f40" />

In the code, the dataset object that collects the testing data has settings regarding whether or not randomization should be performed on the incoming image. Additionally, 
the data from the csv contains data for eight points, while only four are used in the actual project. This was because models that attempted to predict eight points 
performed significantly worse than their four point counterparts.
