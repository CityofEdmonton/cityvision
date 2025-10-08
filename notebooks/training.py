


# Import necessary libraries
import os
import subprocess
from google.cloud import storage
from ultralytics import YOLO
import yaml
import shutil
import random
import cv2
import numpy as np
import zipfile
import yaml
from typing import Union, Tuple
import sys
from ultralytics import settings
import mlflow



# In[27]:


# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "."  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = os.getcwd()  # the current directory where the data will be downloaded
# MLflow Configuration
MLFLOW_EXPERIMENT_NAME = "YOLO_Object_Detection_Training"
MLFLOW_RUN_NAME = "yolov11m_run"


# In[28]:


# YOLO Model Configuration
IMG_SIZE = 640
BATCH_SIZE = 64
DEVICE = 0
EPOCHS = 50


# In[29]:


def download_data_from_gcs(bucket_name: str, gcs_path: str, local_dir: str) -> None:
    """
    Downloads a directory and its contents from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_path (str): The path to the directory in GCS (e.g., "datasets/my_data/").
        local_dir (str): The local directory to save the downloaded files.
    """
    # if the gcs_path is just ".", we want to list all blobs in the bucket
    if gcs_path == "" or gcs_path == ".":
        prefix = ""
    else:
        # Ensure the prefix ends with a '/' to treat it as a directory
        prefix = gcs_path.rstrip("/") + "/"

    print(
        f"Attempting to download data from gs://{bucket_name}/{gcs_path} to {local_dir}"
    )

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)

        blobs = bucket.list_blobs(
            prefix=prefix, delimiter="/"
        )  # List all blobs with the given prefix
        downloaded_count = 0
        for blob in blobs:
            # Check if the blob is a .zip file and not a directory itself
            if blob.name.endswith(".zip"):
                # Construct local file path, taking only the base file name
                local_file_name = os.path.basename(blob.name)
                local_file_path = os.path.join(local_dir, local_file_name)

                print(f"Downloading {blob.name} to {local_file_path}")
                blob.download_to_filename(local_file_path)
                downloaded_count += 1
        if downloaded_count == 0:
            print(
                f"No zipped files found or downloaded from gs://{bucket_name}/{gcs_path}. "
                f"Please check bucket name and GCS path, and ensure there are .zip files present."
            )
        else:
            print(f"Successfully downloaded {downloaded_count} zipped files from GCS.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Error downloading data from GCS: {e}")
        print("Please ensure your Google Cloud credentials are set up correctly.")
        print(
            "You can use `gcloud auth application-default login` for local development or set the "
            "`GOOGLE_APPLICATION_CREDENTIALS` environment variable for service accounts."
        )
        exit(1)  # Exit if data download fails


# In[30]:


# unzipping files
def unzipDataset(folderPath: str) -> None:
    """
    Unzips all .zip files in the specified folder.
    Args:
        folderPath (str): The path to the folder containing .zip files.
    """
    files = os.listdir(folderPath)
    for file in files:
        if not file.endswith(".zip"):
            continue
        filename = file.split(".")[0]
        fullPath = os.path.join(folderPath, file)
        if os.path.isfile(fullPath):
            with zipfile.ZipFile(fullPath, "r") as zip_ref:
                zip_ref.extractall(os.path.join(folderPath, filename))

        os.remove(fullPath)
    return filename


# In[31]:


# --- 2. Function to Train YOLO Model ---
def train_yolo_model(
    data_yaml_path: str,
    model: YOLO,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    mlflow_tracking: bool = False,
    hsv_h_range: Union[float, Tuple[float, float]] = 0.05,
    hsv_s_range: Union[float, Tuple[float, float]] = 0.5,
    hsv_v_range: Union[float, Tuple[float, float]] = 0.3,
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
        hsv_h_range (Union[float, Tuple[float, float]]): Hue augmentation range.
        hsv_s_range (Union[float, Tuple[float, float]]): Saturation augmentation range.
        hsv_v_range (Union[float, Tuple[float, float]]): Brightness augmentation range.
    """
    try:
        print(f"\n--- Starting YOLO Model Training with {model} ---")
        if mlflow_tracking:
            print("MLflow logging is enabled.")
            settings.update({"mlflow": True})
             # Log hyperparameters to MLflow
            mlflow.log_params({
                "epochs": epochs,
                "img_size": img_size,
                "batch_size": batch_size,
                "device": device,
                "model_type": model.model.yaml.get('yaml_file', 'yolo11m'),
                "hsv_h": hsv_h_range,
                "hsv_s": hsv_s_range,
                "hsv_v": hsv_v_range,
                "data_yaml": data_yaml_path,
            })
        else:
            print("MLflow logging is disabled. Training will not log to MLflow.")
        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            device=device,
            val=False,  # Enable validation during training
            # --- ONLY HSV Augmentations ---
            hsv_h=hsv_h_range,  # Hue augmentation (randomly adjusted within +/- hsv_h_range)
            hsv_s=hsv_s_range,  # Saturation augmentation (randomly adjusted within +/- hsv_s_range)
            hsv_v=hsv_v_range,  # Brightness augmentation (randomly adjusted within +/- hsv_v_range)
            # --- Disable Other Augmentations ---
            degrees=0.0,  # Image rotation
            translate=0.0,  # Image translation
            scale=0.0,  # Image scaling
            shear=0.0,  # Image shearing
            perspective=0.0,  # Image perspective transformation
            flipud=0.0,  # Flip image upside down
            fliplr=1.0,  # Flip image left-right
            mosaic=0.0,  # Disable mosaic augmentation
            mixup=0.0,  # Disable mixup augmentation
            copy_paste=0.0,  # Disable copy-paste augmentation
            project="ultralytics_yolo_project",
            name="yolov11l_run",
            # auto_augment=None # Ensure auto_augment is not overriding
        )
        
        
        print("\n--- Training Complete! ---")
        save_dir = model.trainer.save_dir  # Directory where results are saved
        print(f"Results saved to: {save_dir}")
        print("You can find the best.pt and last.pt models in this directory.")
        if mlflow_tracking:
                print("\n--- Logging Model and Artifacts to MLflow ---")
                
                # Log the entire 'runs/...' directory as an artifact
                mlflow.log_artifacts(save_dir, artifact_path="yolo_output")

                # Log the best model using mlflow.log_model
                best_model_path = os.path.join(save_dir, "weights", "best.pt")
                if os.path.exists(best_model_path):
                    # Using log_artifact for the .pt file
                    mlflow.log_artifact(best_model_path, artifact_path="models")
                    print(f"Logged best model: {best_model_path}")

                # Note: For metric logging, Ultralytics usually integrates this
                # when MLflow is active (via the `mlflow` environment variable
                # or by detection of an active run). If manual logging is needed:
                # results.results_dict contains final metrics (e.g., 'metrics/mAP50(B)')
                # for metric_name, metric_value in results.results_dict.items():
                #     mlflow.log_metric(metric_name.replace('/', '_'), metric_value)

                print("MLflow logging complete.")

    except Exception as e:
        print(f"Error during YOLO model training: {e}")
        print(
            "Please ensure Ultralytics is installed (`pip install ultralytics`) "
            "and your dataset `data.yaml` is correctly configured."
        )
        exit(1)


# In[32]:


def train_with_data_in_cloud():
    # 1. Download data from GCS
    download_data_from_gcs(GCS_BUCKET_NAME, GCS_DATA_PATH, LOCAL_DATA_DIR)
    # 2. Unzip the downloaded files
    dataset_name = unzipDataset(LOCAL_DATA_DIR)
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_name, "data.yaml")
    # 3. Train the YOLO model
    print("the path from yaml is ", yaml_path)
    with mlflow.start_run(experiment_id=mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME).experiment_id, run_name=MLFLOW_RUN_NAME):
        model = YOLO("yolo11l.pt")
        train_yolo_model(
            data_yaml_path=yaml_path,
            model=model,
            epochs=EPOCHS,
            img_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            mlflow_tracking=True,  # Enable MLflow logging
        )
    return model
def train_with_data_locally(dataset_location):
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_location, "data.yaml")
    # 3. Train the YOLO model
    model = YOLO("yolo11l.pt")
    train_yolo_model(
        data_yaml_path=yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )
    return model


# In[33]:


# --- Main Execution Flow ---
if __name__ == "__main__":

    # train model based on where the data is
    model = train_with_data_in_cloud()
    #model = train_with_data_locally(os.path.join("yolo_dataset", "2025-1-07_len_14200"))
