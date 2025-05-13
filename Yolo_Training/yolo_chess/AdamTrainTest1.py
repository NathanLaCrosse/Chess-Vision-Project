import os
import sys
import torch
from ultralytics import YOLO

# Allow duplicate OpenMP libraries (use with caution)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # model & data paths
    model_path = os.path.join(base_dir, "yolo11m.pt")
    yaml_path = os.path.join(base_dir, "chess.yaml")

    # quick existence checks
    if not os.path.isfile(model_path):
        sys.exit(f"ERROR: weight file not found at {model_path}")
    if not os.path.isfile(yaml_path):
        sys.exit(f"ERROR: data YAML not found at {yaml_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    # load model
    model = YOLO(model_path)

    # TRAIN
    model.train(
        data=yaml_path,
        epochs=200,  # full training
        batch=8,
        device=device,
        lr0=0.01, lrf=0.1,
        momentum=0.9,
        weight_decay=0.005,
        project="Yolo_Training",
        name="yolo_chess2",
        exist_ok=True
    )

    # PREDICT on validation folder
    print("\n--- Running inference on validation set ---")
    results = model.predict(
        source=os.path.join(base_dir, "dataset/images/val"),
        device=device,
        conf=0.25,
        save=True
    )


    # results is a list of Results objects; they all share the same save_dir
    first = results[0]
    print(f"Saved inference results to {first.save_dir}")


if __name__ == "__main__":
    main()
