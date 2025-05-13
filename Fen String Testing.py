import Visualizer as v
from pathlib import Path

folder_dir = Path("Data/Test Images")

# For each file in the test images folder, predict results and display them.
for file_dir in folder_dir.iterdir():
    print(f"Results for image at {file_dir}")
    v.visualize_final_results(file_dir, conf=0.5)
    print("\n\n\n")