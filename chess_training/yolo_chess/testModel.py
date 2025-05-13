
from ultralytics import YOLO
YOLO('yolo11m.pt')  # this will auto-download the official medium model





# 1. Point at your trained checkpoint
model = YOLO('Chess-Vision-Project/chess_training/yolo_chess/weights/best.pt')

# 2. Run inference on a single image
results = model.predict(
    source='Data/Test Images/1.jpg',
    imgsz=640,
    conf=0.6,                             # confidence threshold
    save=True                              # save annotated images to disk
)

# 3. Display the first result in a window
results[0].show()


boxes = results[0].boxes       # Boxes object
df    = boxes.dataframe()      # pandas DataFrame with xmin, ymin, xmax, ymax, confidence, class
print(df)
