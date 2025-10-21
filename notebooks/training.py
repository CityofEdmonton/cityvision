# Import necessary libraries
from ultralytics import YOLO
import os
from typing import Union, Tuple
from ultralytics import settings
from utils import download_data_from_gcs, unzipDataset


# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "."  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = os.getcwd()  # the current directory where the data will be downloaded
MLFLOW_EXPERIMENT_NAME = "_yolo_aug"

# YOLO Model Configuration
IMG_SIZE = 640
BATCH_SIZE = 64
DEVICE = 0
EPOCHS = 200
FREEZE_LAYERS = 10
LEARNING_RATE = 2e-4
ML_FLOW_TRACKING = True
AUGMENTATION = {
    "hsv_h_range": 0.015,  # Hue augmentation (randomly adjusted within +/- 0.015)
    "hsv_s_range": 0.3,  # Saturation augmentation (randomly adjusted within +/- 0.3)
    "hsv_v_range": 0.3,  # Brightness augmentation (randomly adjusted within +/- 0.3)
    "fliplr": 0.5,  # Flip image left-right with a probability of 0.5
    "degrees": 5,  # Image rotation
}


# --- 2. Function to Train YOLO Model ---
def train_yolo_model(
    data_yaml_path: str,
    model: YOLO,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    mlflow_tracking: bool = False,
    augmentations: dict = None,
) -> None:
    """
    Trains a YOLO model with specified parameters.
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        model (YOLOModel): YOLO model instance to train.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.
        batch_size (int): Batch size for training.
        device (str): Device to train on (e.g., '0' for GPU, 'cpu').
        mlflow_tracking (bool): Whether to enable MLflow tracking.
        augmentations (dict): Dictionary containing augmentation parameters.
    """
    try:
        print(f"\n--- Starting YOLO Model Training with {model} ---")
        if mlflow_tracking:
            print("\n--- MLflow Logging is Enabled ---")
            # Set up MLflow experiment
            settings.update({"mlflow": True})

        else:
            print("MLflow logging is disabled. Training will not log to MLflow.")
        # Train the model
        if augmentations is None:
            hsv_h_range = 0
            hsv_s_range = 0
            hsv_v_range = 0
            fliplr = 0
            degrees = 0
        else:
            hsv_h_range = augmentations.get("hsv_h_range", 0)
            hsv_s_range = augmentations.get("hsv_s_range", 0)
            hsv_v_range = augmentations.get("hsv_v_range", 0)
            fliplr = augmentations.get("fliplr", 0)
            degrees = augmentations.get("degrees", 0)

        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            freeze=FREEZE_LAYERS,  # Freeze the first 10 layers
            lr0=LEARNING_RATE,
            dropout=0.2,
            cos_lr=False,
            plots=True,
            val=True,
            # --- ONLY HSV Augmentations ---
            hsv_h=hsv_h_range,  # Hue augmentation (randomly adjusted within +/- hsv_h_range)
            hsv_s=hsv_s_range,  # Saturation augmentation (randomly adjusted within +/- hsv_s_range)
            hsv_v=hsv_v_range,  # Brightness augmentation (randomly adjusted within +/- hsv_v_range)
            fliplr=fliplr,  # Flip image left-right
            # --- Disable Other Augmentations ---
            degrees=degrees,  # Image rotation
            translate=0.0,  # Image translation
            scale=0.0,  # Image scaling
            shear=0.0,  # Image shearing
            perspective=0.0,  # Image perspective transformation
            flipud=0.0,  # Flip image upside down
            mosaic=0.0,  # Disable mosaic augmentation
            mixup=0.0,  # Disable mixup augmentation
            copy_paste=0.0,  # Disable copy-paste augmentation
            project="ultralytics_yolo_project" + MLFLOW_EXPERIMENT_NAME,
            name="yolov11l_run",
            # auto_augment=None # Ensure auto_augment is not overriding
        )

        print("\n--- Training Complete! ---")
        save_dir = model.trainer.save_dir  # Directory where results are saved
        print(f"Results saved to: {save_dir}")
        # write this directory to the console
        print(f"Results saving to console ...")
        destination_prefix = "Models"
        for root, _, files in os.walk(save_dir):
            for file in files:
                print(f"Uploading file: {file}")
                local_file_path = os.path.join(root, file)
                # Create a GCS destination path that maintains the folder structure
                relative_path = os.path.relpath(local_file_path, save_dir)
                folder = root.split("/")[-1]
                gcs_path = os.path.join(
                    destination_prefix, folder, relative_path
                ).replace(
                    "\\", "/"
                )  # Use forward slashes
                print(f"GCS Path: {gcs_path}")
                from google.cloud import storage

                storage_client = storage.Client()
                bucket = storage_client.bucket("open-cityvision")
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_file_path)

        print("You can find the best.pt and last.pt models in this directory.")

    except Exception as e:
        print(f"Error during YOLO model training: {e}")
        print(
            "Please ensure Ultralytics is installed (`pip install ultralytics`) "
            "and your dataset `data.yaml` is correctly configured."
        )
        exit(1)


def train_with_data_in_cloud():
    # 1. Download data from GCS
    download_data_from_gcs(GCS_BUCKET_NAME, GCS_DATA_PATH, LOCAL_DATA_DIR)
    # 2. Unzip the downloaded files
    dataset_name = unzipDataset(LOCAL_DATA_DIR)
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_name, "data.yaml")
    # 3. Train the YOLO model
    print("the path from yaml is ", yaml_path)
    augmentation = AUGMENTATION
    model = YOLO("yolo11l.pt")
    train_yolo_model(
        data_yaml_path=yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        mlflow_tracking=ML_FLOW_TRACKING,  # Enable MLflow logging
    )
    return model


def train_with_data_locally(dataset_location):
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_location, "data.yaml")
    # 3. Train the YOLO model
    model = YOLO("yolo11l.pt")
    augmentation = AUGMENTATION
    train_yolo_model(
        data_yaml_path=yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        mlflow_tracking=ML_FLOW_TRACKING,
        augmentations=augmentation,
    )
    return model


# --- Main Execution Flow ---
if __name__ == "__main__":
    print("Starting the training script...")
    # train model based on where the data is
    # model = train_with_data_in_cloud()
    model = train_with_data_locally("2025-10-09_len_14200")
