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

# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "Dataset/"  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = "yolo_dataset/"  # Local directory to download data to

# YOLO Model Configuration
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = "cpu"  # 0 for GPU (if available), 'cpu' for CPU
EPOCHS = 10


def download_data_from_gcs(bucket_name: str, gcs_path: str, local_dir: str) -> None:
    """
    Downloads a directory and its contents from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_path (str): The path to the directory in GCS (e.g., "datasets/my_data/").
        local_dir (str): The local directory to save the downloaded files.
    """
    print(
        f"Attempting to download data from gs://{bucket_name}/{gcs_path} to {local_dir}"
    )

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)

        blobs = bucket.list_blobs(
            prefix=gcs_path
        )  # List all blobs with the given prefix
        downloaded_count = 0
        for blob in blobs:
            # Skip blobs that represent directories themselves (ending with '/')
            # And filter for files that end with '.zip'
            if not blob.name.endswith("/") and blob.name.endswith(".zip"):
                # Construct local file path, preserving relative directory structure if any
                local_file_name = os.path.basename(blob.name)
                local_file_path = os.path.join(local_dir, local_file_name)

                blob.download_to_filename(local_file_path)
                downloaded_count += 1
            else:
                print(f"Skipping non-zipped file or directory: {blob.name}")

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


def unzipDataset(folderPath: str) -> None:
    """
    Unzips all .zip files in the specified folder.
    Args:
        folderPath (str): The path to the folder containing .zip files.
    """
    files = os.listdir(folderPath)
    for file in files:
        filename = file.split(".")[0]
        fullPath = os.path.join(folderPath, file)
        if os.path.isfile(fullPath):
            with zipfile.ZipFile(fullPath, "r") as zip_ref:
                zip_ref.extractall(folderPath)

        os.remove(fullPath)


def update_traintxtfile(train_filename: str, individual_dataset_part_base: str) -> None:
    """
    Updates the train.txt file with paths relative to the master_data.yaml's 'path'.

    Args:
        train_filename (str): The name of the train file (e.g., 'Train.txt').
        individual_dataset_part_base (str): The base path for the individual dataset part.
    """
    if train_filename:
        # Construct the full path to the train file
        print(
            f"Updating train file: {train_filename} in {individual_dataset_part_base}"
        )
        train_file_path = os.path.join(individual_dataset_part_base, train_filename)

        # Read the existing train file
        with open(train_file_path, "r") as file:
            lines = file.readlines()

        # Update each line to be relative to the master_data.yaml's 'path'
        updated_lines = [
            os.path.join(individual_dataset_part_base, line.strip()) + "\n"
            for line in lines
        ]

        # Write the updated lines back to the train file
        with open(train_file_path, "w") as file:
            file.writelines(updated_lines)


def manage_Image_Label_Folder(folder_path: str) -> None:
    """
    Ensures the folder structure for images and labels is correct.

    Args:
        folder_path (str): The path to the folder containing images and labels.
    """
    # if the folder contains the image and labels folder, then create a data folder and move these there
    if os.path.exists(os.path.join(folder_path, "images")) and os.path.exists(
        os.path.join(folder_path, "labels")
    ):
        data_folder = os.path.join(folder_path, "data")
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        # Move images and labels to the data folder
        shutil.move(
            os.path.join(folder_path, "images"), os.path.join(data_folder, "images")
        )
        shutil.move(
            os.path.join(folder_path, "labels"), os.path.join(data_folder, "labels")
        )
        print(f"Moved images and labels to {data_folder}")


def generate_master_yaml(dataset_root_folder, output_yaml_name="master_data.yaml"):
    """
    Generates a master data.yaml file for YOLOv9 training by combining
    information from individual data.yaml files found in subfolders.

    Args:
        dataset_root_folder (str): The path to the root folder containing
                                   subfolders, each with its own data.yaml.
        output_yaml_name (str): The name of the master YAML file to create.
    """
    master_train_paths = []
    master_val_paths = []  # Initialize for potential validation paths
    master_names = {}

    # Get the absolute path for the master YAML file
    master_output_path = os.path.join(dataset_root_folder, output_yaml_name)

    # We'll use the root folder itself as the 'path' in the master_data.yaml
    # This means all individual train/val paths should be relative to dataset_root_folder.

    # Iterate through direct subfolders in the dataset_root_folder
    subfolders = [
        d
        for d in os.listdir(dataset_root_folder)
        if os.path.isdir(os.path.join(dataset_root_folder, d))
    ]

    if not subfolders:
        print(
            f"No subfolders found in '{dataset_root_folder}'. Please ensure your dataset parts are in subfolders."
        )
        return

    # Try to load class names from the first found data.yaml
    first_data_yaml_found = False
    print(
        f"Looking for 'data.yaml' files in subfolders of '{dataset_root_folder}' to extract class names..."
    )
    # -----Getting the names section-----
    for folder in subfolders:
        print(f"Checking folder: {folder}")
        individual_data_yaml_path = os.path.join(
            dataset_root_folder, folder, "data.yaml"
        )
        if os.path.exists(individual_data_yaml_path):
            try:
                with open(individual_data_yaml_path, "r") as f:
                    individual_yaml_data = yaml.safe_load(f)
                master_names = individual_yaml_data.get("names", {})
                if not master_names:
                    print(
                        f"Warning: 'names' section not found in '{individual_data_yaml_path}'. Please ensure your data.yaml files define class names."
                    )
                else:
                    print(f"Class names loaded from: {individual_data_yaml_path}")
                first_data_yaml_found = True
                break  # Found names, break out
            except yaml.YAMLError as e:
                print(
                    f"Error parsing YAML file '{individual_data_yaml_path}': {e}. Skipping for names."
                )
            except FileNotFoundError:
                # This should not happen if os.path.exists was true, but as a safeguard
                print(
                    f"Error: data.yaml file '{individual_data_yaml_path}' disappeared unexpectedly. Skipping."
                )
    if not first_data_yaml_found:
        print(
            "Error: No 'data.yaml' found in any subfolder to extract class names. Cannot generate master YAML."
        )
        return

    # Collect train and val paths
    for folder in subfolders:
        full_folder_path = os.path.join(dataset_root_folder, folder)
        manage_Image_Label_Folder(
            full_folder_path
        )  # Ensure the folder is managed correctly
        print(f"Processing folder: {folder}")
        individual_data_yaml_path = os.path.join(
            dataset_root_folder, folder, "data.yaml"
        )
        if not os.path.exists(individual_data_yaml_path):
            print(
                f"Warning: '{individual_data_yaml_path}' not found. Skipping folder '{folder}'."
            )
            continue
        try:
            with open(individual_data_yaml_path, "r") as f:
                data = yaml.safe_load(f)

            # Determine the base path for this individual YAML's relative paths
            # According to your example, 'path: .' means Train.txt is relative to the folder itself.
            individual_yaml_base_path_in_yaml = data.get(
                "path", "."
            )  # This will be '.'

            # This is the actual directory where 'Train.txt' is expected to be found
            # relative to the dataset_root_folder
            individual_dataset_part_base = os.path.join(folder)

            # --- Handle Training Paths ---
            train_filename = data.get("Train")  # e.g., 'Train.txt'
            full_train_path = os.path.join(full_folder_path, train_filename)
            update_traintxtfile(
                full_train_path, individual_dataset_part_base
            )  # Update the train.txt file
            if train_filename:
                # Construct path relative to the master_data.yaml's 'path' (dataset_root_folder)
                # which means it's relative to the 'individual_dataset_part_base' (e.g., 'part1')
                combined_train_path = os.path.join(
                    individual_dataset_part_base, train_filename
                )
                master_train_paths.append(combined_train_path)
            else:
                print(
                    f"Warning: 'Train' path not found in '{individual_data_yaml_path}'. Skipping training data from this part."
                )

            # --- Handle Validation Paths (look for 'Val' or 'val') ---
            val_filename = data.get("Val") or data.get("val")  # Check both cases
            if val_filename:
                # Construct path relative to the master_data.yaml's 'path'
                combined_val_path = os.path.join(
                    individual_dataset_part_base, val_filename
                )
                master_val_paths.append(combined_val_path)
            # TODO: Handle case where no validation path is found
            else:
                print(
                    f"Info: No 'Val' or 'val' path found in '{individual_data_yaml_path}'. "
                    "This part will not contribute validation data to the master YAML."
                )

        except yaml.YAMLError as e:
            print(
                f"Error parsing YAML file '{individual_data_yaml_path}': {e}. Skipping folder '{folder}'."
            )
        except Exception as e:
            print(
                f"An unexpected error occurred processing '{individual_data_yaml_path}': {e}. Skipping folder '{folder}'."
            )
        # Construct the master YAML data dictionary
    master_data_content = {
        "path": "../yolo_dataset",  # Path from where master.yaml is called
        "train": master_train_paths,
        "nc": len(master_names),
        "names": master_names,
    }
    if master_val_paths:
        master_data_content["val"] = master_val_paths
    else:
        print(
            "\nNote: No validation paths were found in any individual data.yaml files. "
        )
        print("Including a train path to the master YAML.")
        master_val_paths.append(
            master_train_paths[-1]
        )  # Use the last train path as a fallback
        master_train_paths.pop()  # Remove the last train path since we are using it for validation
        master_data_content["val"] = (
            master_val_paths  # Add the fallback validation path
        )

    # Write the master YAML file
    try:
        with open(master_output_path, "w") as f:
            yaml.dump(master_data_content, f, default_flow_style=False, sort_keys=False)
        print(f"\nMaster YAML file '{master_output_path}' generated successfully!")
        print("\nContents of generated master_data.yaml:")
        print(yaml.dump(master_data_content, default_flow_style=False, sort_keys=False))
    except Exception as e:
        print(f"Error writing master data.yaml to '{master_output_path}': {e}")
        # Optionally delete partially written file if error occurs
        if os.path.exists(master_output_path):
            os.remove(master_output_path)
    return master_output_path


def train_yolo_model(
    data_yaml_path: str,
    model: YOLO,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
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
        hsv_h_range (Union[float, Tuple[float, float]]): Hue augmentation range.
        hsv_s_range (Union[float, Tuple[float, float]]): Saturation augmentation range.
        hsv_v_range (Union[float, Tuple[float, float]]): Brightness augmentation range.
    """
    try:
        print(f"\n--- Starting YOLO Model Training with {model} ---")
        # Train the model
        # get the data_yml_path_files
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
            fliplr=0.0,  # Flip image left-right
            mosaic=0.0,  # Disable mosaic augmentation
            mixup=0.0,  # Disable mixup augmentation
            copy_paste=0.0,  # Disable copy-paste augmentation
            # auto_augment=None # Ensure auto_augment is not overriding
        )

        print("\n--- Training Complete! ---")
        print(f"Results saved to: {model.trainer.save_dir}")
        print("You can find the best.pt and last.pt models in this directory.")

    except Exception as e:
        print(f"Error during YOLO model training: {e}")
        print(
            "Please ensure Ultralytics is installed (`pip install ultralytics`) "
            "and your dataset `data.yaml` is correctly configured."
        )
        exit(1)


if __name__ == "__main__":
    # 0. Clean up
    if os.path.exists(LOCAL_DATA_DIR):
        print(f"Cleaning up: Removing existing local data directory {LOCAL_DATA_DIR}")
        shutil.rmtree(LOCAL_DATA_DIR)

    # 1. Download data from GCS
    download_data_from_gcs(GCS_BUCKET_NAME, GCS_DATA_PATH, LOCAL_DATA_DIR)

    # 2. Unzip the downloaded files
    unzipDataset(LOCAL_DATA_DIR)

    # 3. Create master data.yaml
    master_yaml_path = generate_master_yaml(LOCAL_DATA_DIR, "master_data.yaml")

    if not os.path.exists(master_yaml_path):
        print(
            f"Error: {master_yaml_path} not found. Please ensure your GCSing and unzipping steps were successful."
        )
        exit(1)

    # Verify data.yaml content
    try:
        with open(master_yaml_path, "r") as f:
            data_config = yaml.safe_load(f)
            print("\n--- Loaded data.yaml configuration ---")
            print(yaml.dump(data_config, indent=2))
            # Ensure paths in data.yaml are relative to the LOCAL_DATA_DIR
            # or are absolute paths to the downloaded data.
            # Example: train: images/train/
            #          val: images/val/
            #          names: [...]
    except Exception as e:
        print(f"Error loading data.yaml: {e}")
        exit(1)

    # 3. Train the YOLO model

    model = YOLO("yolo11m.pt")
    train_yolo_model(
        data_yaml_path=master_yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    # --- Clean up ---
    print(f"\nCleaning up: Removing local data directory {LOCAL_DATA_DIR}")
    shutil.rmtree(LOCAL_DATA_DIR)
