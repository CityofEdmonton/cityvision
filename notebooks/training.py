# Import necessary libraries
from ultralytics import YOLO
import yaml
from collections import Counter
import os
import os
import sys
from typing import Union, Tuple
from ultralytics import settings
from utils import download_data_from_gcs, unzipDataset, analyze_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.build import InfiniteDataLoader, build_dataloader
from ultralytics.utils import DEFAULT_CFG
from torch.utils.data import WeightedRandomSampler
from ultralytics.utils.torch_utils import torch_distributed_zero_first
import torch.nn as nn
import numpy as np
import torch
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import  make_anchors
from typing import Any, Union, Tuple, Dict
from ultralytics.utils.loss import FocalLoss

from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "."  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = os.getcwd()  # the current directory where the data will be downloaded
MLFLOW_EXPERIMENT_NAME = "_nov_18_yolo11n_loss_classweights"

# YOLO Model Configuration
IMG_SIZE = (384, 576) # TODO: change to imgsz=(384, 576)
BATCH_SIZE = 128
DEVICE = 0
EPOCHS = 1
FREEZE_LAYERS = 10
LEARNING_RATE = 0.0001
COSLR = False
ML_FLOW_TRACKING = True
REGULARIZATION_WEIGHT = 0.001
DROPOUT = 0.1
MODEL_TYPE = "yolo11n.pt"
WORKERS = 0  # Number of data loading workers (0 means however many cores are available)
OPTIMIZER = "Adam"  # setting optimizer ot Adam to make sure the lr is set correctly
ALPHA_FOR_SAMPLER = 0.7
LABEL_SMOOTHING_FACTOR = 0.1 # Set to 0 to disable label smoothing, or a value in [0, 1] to enable
PATIENCE = 400
AUGMENTATION = {
    # --- Photometric (HSV) Augmentations ---
    "hsv_h": 0.015,         # Hue augmentation (+/- 0.015)
    "hsv_s": 0.7,           # Saturation augmentation (+/- 0.7)
    "hsv_v": 0.6,           # Brightness/Value augmentation (+/- 0.4)

    # --- Geometric Augmentations (Requires BBox Transformation) ---
    "degrees": 180.0,       # Image rotation (+/- degrees). Set to 1.0 - 5.0 if needed.
    "translate": 0.3,       # Image translation (+/- fraction of image size)
    "scale": 0.5,           # Image scaling (zoom out 0.5x to zoom in 1.5x)
    "shear": 5,           # Image shear (+/- degrees). Set to 1.0 - 5.0 if needed.
    "perspective": 0.0001	,     # Perspective transform (random fraction). Set to 0.001 if needed.
    "flipud": 1,          # Flip image Up-Down (Probability). Set to 0.1 for general tasks.
    "fliplr": 1,          # Flip image Left-Right (Probability)
    # "bgr": 0.5,             # Convert image to BGR color space (Probability)
    # --- Compositional Augmentations (Often applied together) ---
    # "mosaic": 0.5,          # Combine 4 images into 1 (Probability)
    # "mixup": 0.3,           # Blend 2 images and labels (Probability)
    "cutmix": 1,          # Cut a patch from one image and paste to another (Probability)
    "copy_paste": 1,      #see mixupo# Copy objects from one image and paste to another (Probability)
}

CLASS_WEIGHTS = True # Make this to None to disable class weights

def get_class_weights(data_yaml_path):
    class_counts = analyze_yolo_dataset(data_yaml_path)[0]
    total = sum(class_counts.values())
    nc = len(class_counts)
    class_weights = [1.0 for _ in range(nc)]

    if total > 0:
        for i in range(nc):
            count = class_counts.get(i, 0)
            class_weights[i] = 1.0 / (count + 1e-6)
        s = sum(class_weights)
        if s > 0:
            class_weights = [cw * nc / s for cw in class_weights]
        else:
            class_weights = [1.0 for _ in range(nc)]
    return class_weights

def taper_augmentations(trainer, start_ratio=0.70):
    """
    Tapers the augmentation strength during training.
    Args:
        trainer: The training object containing epoch and hyp attributes.
        start_ratio: The ratio of epochs after which to start tapering.
    
    """
    r = trainer.epoch / max(1, trainer.epochs)
    if r >= start_ratio:
        H = trainer.hyp  # training hyper-params dict
        # zero out strong regs to let model fit real distribution
        for k in AUGMENTATION.keys():
            if H.get(k, 0) > 0:
                H[k] = 0.0

def on_train_epoch_start(trainer):
    """Callback to taper augmentations at the start of each training epoch."""
    taper_augmentations(trainer, start_ratio=0.90)

callbacks = {
    "on_train_epoch_start": on_train_epoch_start,
}

def train_yolo_model(
    data_yaml_path: str,
    model: YOLO,
    epochs: int,
    img_size: int,
    batch_size: int,
    device: str,
    mlflow_tracking: bool = False,
    augmentations: dict = None,
    albumentations_transforms: Any = None,
    callbacks: dict = callbacks,
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
        # Manually create the overrides dictionary using all your custom arguments
        custom_overrides = {
            'data': data_yaml_path,
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': device,
            'freeze': FREEZE_LAYERS,
            'lr0': LEARNING_RATE,
            'dropout': DROPOUT,
            'weight_decay': REGULARIZATION_WEIGHT,
            'workers': WORKERS,
            'optimizer': OPTIMIZER,
            'val': True,
            'cos_lr': COSLR,
            'plots': True,
            'project': "ultralytics_yolo_project" + MLFLOW_EXPERIMENT_NAME,
            'name': "yolov11n_run",
            'patience': PATIENCE,
            'label_smoothing': LABEL_SMOOTHING_FACTOR,
            'class_weights': CLASS_WEIGHTS,

        }
        if augmentations:
            custom_overrides.update(augmentations)

        model.overrides.update(custom_overrides)
        for i in callbacks:
            model.add_callback(i, callbacks[i])
        
        # Note: Loss verification will happen automatically in _setup_train() when training starts
        # The verification prints will show up when _setup_train() is called
        print("\n--- Starting Training (Custom Loss Verification will occur during setup) ---", flush=True)
        sys.stdout.flush()
        
        results = model.train()
    
        
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
                from datetime import date
                todays_date = date.today().strftime("%Y-%m-%d")
                folder = todays_date +  MLFLOW_EXPERIMENT_NAME
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
    return results

def train_with_data_in_cloud():
    """
    This function downloads data from GCS, unzips it, and trains the YOLO model.
    Returns:
        model (YOLO): Trained YOLO model instance.
    """
    
    # 1. Download data from GCS
    download_data_from_gcs(GCS_BUCKET_NAME, GCS_DATA_PATH, LOCAL_DATA_DIR)
    # 2. Unzip the downloaded files
    dataset_name = unzipDataset(LOCAL_DATA_DIR)
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_name, "data.yaml")
    # 3. Train the YOLO model
    print("the path from yaml is ", yaml_path)
    augmentation = AUGMENTATION
    model = YOLO(MODEL_TYPE)
    train_yolo_model(
        data_yaml_path=yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        mlflow_tracking=ML_FLOW_TRACKING,  # Enable MLflow logging
        callbacks=callbacks,
    )
    return model

def train_with_data_locally(dataset_location):
    """
    This function trains the YOLO model using data located locally. 
    """
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_location, "data.yaml")
    # 3. Train the YOLO model
    model = YOLO(MODEL_TYPE)
    augmentation = AUGMENTATION
    results = train_yolo_model(
        data_yaml_path=yaml_path,
        model=model,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        mlflow_tracking=ML_FLOW_TRACKING,
        augmentations=augmentation,
        callbacks=callbacks,
    )
    return model

# --- Main Execution Flow ---
if __name__ == "__main__":
    if CLASS_WEIGHTS:
        CLASS_WEIGHTS = get_class_weights(data_yaml_path="2025-10-09_len_14200/data.yaml")
    print("Class weights used for training: ", CLASS_WEIGHTS)
    # train model based on where the data is
    # model = train_with_data_in_cloud()
    model = train_with_data_locally("2025-10-09_len_14200")
    #path = r"ultralytics_yolo_projectAug_overloadeed_yolov11l/yolov11n_run/weights/best.pt"
    #yaml_path = r"2025-10-09_len_14200/data.yaml"
    #get_classwise_results(path, yaml_path) 
    

