# Import necessary libraries
from ultralytics import YOLO
import yaml
from collections import Counter
import os
import sys
from typing import Union, Tuple
from ultralytics import settings
from notebooks.utils import download_data_from_gcs, unzipDataset, analyze_yolo_dataset
from typing import Any
import mlflow
from ultralytics.utils import callbacks
from google.cloud import storage
from datetime import date
import math


# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "."  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = os.getcwd()  # the current directory where the data will be downloaded
MLFLOW_EXPERIMENT_NAME = "_nov_24_class_balanced_0_0001"

# YOLO Model Configuration
IMG_SIZE = 640  # TODO: change to imgsz=(384, 576)
BATCH_SIZE = 128
DEVICE = 0
EPOCHS = 30
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
LABEL_SMOOTHING_FACTOR = (
    0.1  # Set to 0 to disable label smoothing, or a value in [0, 1] to enable
)
PATIENCE = 400
AUGMENTATION = {
    # --- Photometric (HSV) Augmentations ---
    "hsv_h": 0.015,  # Hue augmentation (+/- 0.015)
    "hsv_s": 0.7,  # Saturation augmentation (+/- 0.7)
    "hsv_v": 0.6,  # Brightness/Value augmentation (+/- 0.4)
    # --- Geometric Augmentations (Requires BBox Transformation) ---
    "degrees": 180.0,  # Image rotation (+/- degrees). Set to 1.0 - 5.0 if needed.
    "translate": 0.3,  # Image translation (+/- fraction of image size)
    "scale": 0.5,  # Image scaling (zoom out 0.5x to zoom in 1.5x)
    "shear": 5,  # Image shear (+/- degrees). Set to 1.0 - 5.0 if needed.
    "perspective": 0.0001,  # Perspective transform (random fraction). Set to 0.001 if needed.
    "flipud": 1,  # Flip image Up-Down (Probability). Set to 0.1 for general tasks.
    "fliplr": 1,  # Flip image Left-Right (Probability)
    # "bgr": 0.5,             # Convert image to BGR color space (Probability)
    # --- Compositional Augmentations (Often applied together) ---
    # "mosaic": 0.5,          # Combine 4 images into 1 (Probability)
    # "mixup": 0.3,           # Blend 2 images and labels (Probability)
    "cutmix": 1,  # Cut a patch from one image and paste to another (Probability)
    "copy_paste": 1,  # see mixupo# Copy objects from one image and paste to another (Probability)
}

CLASS_WEIGHTS = True  # Make this to None to disable class weights


def get_class_weights(data_yaml_path, beta=0.9999, mode="class_balanced") -> list:
    """
    mode:
      - "class_balanced": Class-Balanced Loss (Cui et al. - https://arxiv.org/pdf/1901.05555)
      - "inverse_freq"  : your original inverse-frequency scheme
    Args:
        data_yaml_path (str): Path to the data.yaml file.
        beta (float): Hyperparameter for Class-Balanced Loss.
        mode (str): Weighting scheme to use ("class_balanced" or "inverse_freq").
    Returns:
        list: A list of class weights.
    How to use:
        CLASS_WEIGHTS = get_class_weights(data_yaml_path="path/to/data.yaml", mode="class_balanced")
    """
    class_counts = analyze_yolo_dataset(data_yaml_path)[0]  # {class_id: count}
    nc = len(class_counts)
    total = sum(class_counts.values())
    class_weights = [1.0 for _ in range(nc)]

    if total == 0:
        return class_weights

    if mode == "class_balanced":
        # Class-Balanced Loss: w_c = (1 - beta) / (1 - beta^{n_c})
        for c in range(nc):
            n_c = class_counts.get(c, 0)

            if n_c > 0:
                effective_num = 1.0 - math.pow(beta, n_c)
                class_weights[c] = (1.0 - beta) / (effective_num + 1e-8)
            else:
                # class not present in dataset; usually safe to give weight 0
                class_weights[c] = 0.0

        mean_w = sum(class_weights) / max(1, nc)
        if mean_w > 0:
            class_weights = [w / mean_w for w in class_weights]

    elif mode == "inverse_freq":
        tmp_weights = []
        for c in range(nc):
            n_c = class_counts.get(c, 0)
            tmp_weights.append(1.0 / (n_c + 1e-6))

        s = sum(tmp_weights)
        if s > 0:
            class_weights = [w * nc / s for w in tmp_weights]
        else:
            class_weights = [1.0 for _ in range(nc)]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return class_weights


def taper_augmentations(trainer, start_ratio=0.70) -> None:
    """
    Tapers the augmentation strength during training.
    Args:
        trainer: The training object containing epoch and hyp attributes.
        start_ratio: The ratio of epochs after which to start tapering.
    Returns:
        None
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
    MLFLOW_EXPERIMENT_NAME: str = MLFLOW_EXPERIMENT_NAME,
    class_weights: Union[list, None] = CLASS_WEIGHTS,
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
        albumentations_transforms (Any): Albumentations transforms to apply.
        callbacks (dict): Dictionary of callback functions for training.
    returns:
        results: The results object returned by the model.train() method.
    How to use:
        results = train_yolo_model(
            data_yaml_path="path/to/data.yaml",
            model=yolo_model,
            epochs=30,
            img_size=640,
            batch_size=16,
            device='0',
            mlflow_tracking=True,
            augmentations={
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 10.0,
                "translate": 0.1,
                "scale": 0.1,
                "shear": 2.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.5,
            },
            callbacks=callbacks,
        )
    """
    try:
        print(f"\n--- Starting YOLO Model Training with {model} ---")
        if class_weights:
            class_weights = get_class_weights(
                data_yaml_path=data_yaml_path, mode="class_balanced"
            )
        if mlflow_tracking:
            print("\n--- MLflow Logging is Enabled ---")
            # Set up MLflow experiment
            settings.update({"mlflow": True})
            # get the tracking uri and experiment name from environment variables
            tracking_uri = os.getenv(
                "MLFLOW_TRACKING_URI", "https://mlflow-test.edmonton.ca/"
            )
            experiment_name = os.getenv(
                "MLFLOW_EXPERIMENT_NAME", "YOLO_Object_Detection_Training"
            )
            run_name = f"{model.model_name}_{MLFLOW_EXPERIMENT_NAME}"
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)

        else:
            print("MLflow logging is disabled. Training will not log to MLflow.")
        # Manually create the overrides dictionary using all your custom arguments
        custom_overrides = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "device": device,
            "freeze": FREEZE_LAYERS,
            "lr0": LEARNING_RATE,
            "dropout": DROPOUT,
            "weight_decay": REGULARIZATION_WEIGHT,
            "workers": WORKERS,
            "optimizer": OPTIMIZER,
            "val": True,
            "cos_lr": COSLR,
            "plots": True,
            "patience": PATIENCE,
            "label_smoothing": LABEL_SMOOTHING_FACTOR,
            "class_weights": class_weights,
        }
        if mlflow_tracking:
            custom_overrides["project"] = experiment_name
            custom_overrides["name"] = run_name
        if augmentations:
            custom_overrides.update(augmentations)

        model.overrides.update(custom_overrides)
        for i in callbacks:
            model.add_callback(i, callbacks[i])

        # Note: Loss verification will happen automatically in _setup_train() when training starts
        # The verification prints will show up when _setup_train() is called
        print(
            "\n--- Starting Training (Custom Loss Verification will occur during setup) ---",
            flush=True,
        )
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
                todays_date = date.today().strftime("%Y-%m-%d")
                folder = todays_date + MLFLOW_EXPERIMENT_NAME
                gcs_path = os.path.join(
                    destination_prefix, folder, relative_path
                ).replace(
                    "\\", "/"
                )  # Use forward slashes
                print(f"GCS Path: {gcs_path}")

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


def train_with_data_in_cloud() -> YOLO:
    """
    This function downloads data from GCS, unzips it, and trains the YOLO model.
    Returns:
        model (YOLO): Trained YOLO model instance.
    How to use:
        model = train_with_data_in_cloud()

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


def train_with_data_locally(dataset_location: str) -> YOLO:
    """
    This function trains the YOLO model using data located locally.
    Args:
        dataset_location (str): Path to the local dataset directory.
    Returns:
        model (YOLO): Trained YOLO model instance.
    How to use:
        model = train_with_data_locally("path/to/local/dataset")
    """
    # get the location of the data.yaml file
    yaml_path = os.path.join(dataset_location, "data.yaml")
    print("the path from yaml is ", yaml_path)
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


def print_validation_results(model_path: str) -> None:
    """
    Prints the validation results in a readable format.
    Args:
        results: The results object returned by the model.val() method.
    How to use:
        print_validation_results("path/to/model.pt")
    """
    model = YOLO(model_path)
    results = model.val()
    print("\n--- Validation Results ---")
    print(f"mAP@0.5: {results.box.map_50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map_50_95:.4f}")
    print(f"Precision: {results.box.precision:.4f}")
    print(f"Recall: {results.box.recall:.4f}")


# --- Main Execution Flow ---
if __name__ == "__main__":
    if CLASS_WEIGHTS:
        CLASS_WEIGHTS = get_class_weights(
            data_yaml_path="2025-10-09_len_14200/data.yaml", mode="class_balanced"
        )
    print("Class weights used for training: ", CLASS_WEIGHTS)
    # train model based on where the data is
    # model = train_with_data_in_cloud()
    # model = train_with_data_locally("2025-10-09_len_14200")
