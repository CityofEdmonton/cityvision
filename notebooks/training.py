# Import necessary libraries
from ultralytics import YOLO
import os
import sys
from typing import Union, Tuple
from ultralytics import settings
from utils import download_data_from_gcs, unzipDataset
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
print(f"DEBUG CHECK: Initial RANK={RANK}, LOCAL_RANK={LOCAL_RANK}")
# --- Configuration ---
GCS_BUCKET_NAME = "open-cityvision"
GCS_DATA_PATH = "."  # Path *within* your GCS bucket to the dataset root
LOCAL_DATA_DIR = os.getcwd()  # the current directory where the data will be downloaded
MLFLOW_EXPERIMENT_NAME = "nano12Nov"

# YOLO Model Configuration
IMG_SIZE = 640
BATCH_SIZE = 128
DEVICE = 0
EPOCHS = 700
FREEZE_LAYERS = 10
LEARNING_RATE = 0.00002
COSLR = False
ML_FLOW_TRACKING = True
REGULARIZATION_WEIGHT = 0.001
DROPOUT = 0.1
MODEL_TYPE = "yolo11n.pt"
WORKERS = 0  # Number of data loading workers (0 means however many cores are available)
OPTIMIZER = "Adam"  # setting optimizer ot Adam to make sure the lr is set correctly
ALPHA_FOR_SAMPLER = 0.7
LABEL_SMOOTHING_FACTOR = 0 # Set to 0 to disable label smoothing, or a value in [0, 1] to enable
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

CLASS_WEIGHTS = False # Make this to None to disable class weights

def make_image_weights(dataset, alpha=0.7):
    """
    Builds per-image weights from class frequencies.
    alpha in [0,1]: 0 = no weighting, 1 = full inverse-frequency.
    Inputs:
        dataset: a YOLO dataset object with .labels and .data["names"]
        alpha: weighting exponent
    Outputs:
        img_w: a torch.DoubleTensor of per-image weights
    """
    ncls = len(dataset.data["names"])
    cls_counts = np.zeros(ncls, dtype=np.int64)
    img_classes = [set() for _ in range(len(dataset))]

    # dataset.labels is a list of dicts with "cls" and "bboxes"
    for i, lab in enumerate(dataset.labels):
        cls_data = lab["cls"]
        # handle when cls data is not numpy array
        if not isinstance(cls_data, np.ndarray):
            cls_data = np.array(cls_data)
        if cls_data.size == 0 or cls_data.ndim == 0:
            continue
        if cls_data.ndim > 1:
            cls_data = cls_data.flatten()
        if len(cls_data):
            classes = cls_data.astype(int).tolist()
            for c in classes:
                cls_counts[c] += 1
            img_classes[i].update(classes)

    # inverse frequency ^ alpha
    inv = 1.0 / np.clip(cls_counts, 1, None)
    inv = inv ** alpha

    # per-image weight: max weight of classes present (aggressive toward rare)
    img_w = np.array([float(inv[list(s)].max() if s else 1.0) for s in img_classes],
                     dtype=np.float32)
    # normalize (not strictly required)
    img_w /= (img_w.mean() + 1e-12)
    return torch.DoubleTensor(img_w)

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

def get_class_counts(dataset):
    """
    Calculates the total instance count for each class across the dataset.
    """
    ncls = len(dataset.data["names"])
    cls_counts = np.zeros(ncls, dtype=np.int64)

    # dataset.labels is a list of dicts with "cls" and "bboxes"
    for lab in dataset.labels:
        cls_data = lab["cls"]
        if not isinstance(cls_data, np.ndarray):
            cls_data = np.array(cls_data)
        if cls_data.size == 0 or cls_data.ndim == 0:
            continue
        if cls_data.ndim > 1:
            cls_data = cls_data.flatten()
        if len(cls_data):
            classes = cls_data.astype(int).tolist()
            for c in classes:
                cls_counts[c] += 1
                
    return cls_counts # Returns a numpy array of [count_c0, count_c1, ...]


def calculate_inverse_frequency_weights(cls_counts):
    """
    Calculates inverse-frequency weights for the loss function. Simple inverse frequency.

    """
    cls_counts = np.maximum(cls_counts, 1)  # Avoid division by zero
    total_instances = np.sum(cls_counts)

    weights = 1.0 / cls_counts

    return weights.tolist()

class SmoothBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """
    A custom BCE loss that implements label smoothing by modifying the target tensor. It also supports class-wise loss weighting.
    Inherits from PyTorch's standard BCE with logits loss.
    """
    def __init__(self, smooth_alpha=0.0, class_weights=None, **kwargs):
        """
        Initializes the SmoothBCEWithLogitsLoss.
        Args:
            smooth_alpha (float): Smoothing factor in [0, 1]. 0 means no smoothing.
            class_weights (list or tensor): Class-wise weights for the loss. Shape [C].
            **kwargs: Additional keyword arguments for nn.BCEWithLogitsLoss.
        """
        super().__init__(**kwargs)
        self.smooth_alpha = smooth_alpha
        self.reduction = self.reduction # inherit reduction method
        if class_weights is not None:
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        print(f"SmoothBCEWithLogitsLoss initialized with smooth_alpha={smooth_alpha} and class_weights={class_weights}")
    def forward(self, input, target):
        """
        Forward pass for the loss computation.
        """
        if self.smooth_alpha > 0:
            if RANK in {-1, 0}:
                print(f"Applying label smoothing with alpha={self.smooth_alpha}")
            # target is the hard label tensor (0 or 1)
            
            target_smoothed = target.clone()

            # Set positive targets to (1.0 - alpha)
            target_smoothed[target == 1] = 1.0 - self.smooth_alpha
            num_classes = input.size(1)
            print("num_classes in smoothing loss:", num_classes)
            # Set negative targets to alpha
            target_smoothed[target == 0] = self.smooth_alpha/max(1, (num_classes - 1))
            
            target = target_smoothed.to(input.dtype)
        if self.class_weights is not None:
            if RANK in {0,-1}:
                print(f"Rank 0: Applying class weights: {self.class_weights}")
            weights_per_anchor = self.class_weights.to(input.device).view(1, -1) # Shape [1, C]
            
            # Standard BCEWithLogitsLoss does not accept a weight for reduction='none'.
            # We must apply the weight manually after the loss is calculated.
            raw_loss = super().forward(input, target) # This returns a loss tensor of shape [N, C]
            if RANK in {-1, 0}:
                # These two print statements will now only execute and print on the main process
                print(f"[Rank 0 Debug] raw_loss shape: {raw_loss.shape}")
                print(f"[Rank 0 Debug] weights_per_anchor shape: {weights_per_anchor.shape}")
            
            # Manual weighted loss: Multiply the loss by the class weight tensor
            # The multiplication broadcasts correctly: [N, C] * [1, C] = [N, C]
            weighted_loss = raw_loss * weights_per_anchor 
            return weighted_loss
        # Calculate loss using the smoothed targets
        return super().forward(input, target)


class CustomDetectionLoss(v8DetectionLoss):
    """
    A custom detection loss that incorporates label smoothing and weighted loss into the classification loss.
    Inherits from the standard v8DetectionLoss.
    """
    
    def __init__(self, model, class_weights=None):
        super().__init__(model)
        
        
        m = model.model[-1]  # Get the Detect() module instance
        self.nc = 9    
        self.cls_loss_log = torch.zeros(self.nc, device=self.device)
        self.cls_counts_log = torch.zeros(self.nc, device=self.device)
        self._call_count = 0  # Counter to limit print frequency
        weight_status = "Enabled" if class_weights is not None else "Disabled"
        print(f"[CustomLoss] Manual Label Smoothing Activated (Alpha: {LABEL_SMOOTHING_FACTOR})")
        print(f"[CustomLoss] Class-Wise Loss Weighting {weight_status}.")
        print(f"[CustomLoss] Class-Wise Loss Logging Enabled for {self.nc} classes.")
        # Inject Custom Loss and Initialize Logging Tensors
        self.bce = SmoothBCEWithLogitsLoss(
            smooth_alpha=LABEL_SMOOTHING_FACTOR,
            class_weights=class_weights, 
            reduction='none' # Must be 'none' to keep the loss tensor [N, C]
        )

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the sum of the loss for box, cls and dfl, and log class-wise loss."""
        
        # Increment call counter
        self._call_count += 1
        
        # Debug print - only on main process, with explicit flushing, and limit frequency
        if RANK in {-1, 0}:
            # Print on first call and every 100 calls
            if self._call_count == 1 or self._call_count % 100 == 0:
                print(f"[CustomDetectionLoss.__call__] Loss function called! Call #{self._call_count}, RANK={RANK}", flush=True)
                sys.stdout.flush()
        
        # Original Setup and Assignment-  same as base class
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1
        )

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Assigner call to find positive samples and target scores/bboxes
        # We need the assignment output (especially fg_mask and target_scores)
        target_labels, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        
        # Classification Loss

        # Calculate classification loss for all samples/classes (tensor of shape [N_anchors, N_classes])
        cls_losses = self.bce(pred_scores, target_scores.to(dtype))
        
        # Debug print for weighted loss calculation
        if RANK in {-1, 0} and (self._call_count == 1 or self._call_count % 100 == 0):
            print(f"[CustomDetectionLoss.__call__] Calculated cls_losses shape: {cls_losses.shape}, fg_mask sum: {fg_mask.sum().item()}", flush=True)
            sys.stdout.flush()
        
        # Get ground truth classes for positive samples 
        gt_classes = target_scores[fg_mask].argmax(dim=1) # [N_positive]
        
        # Accumulate Class-Wise Loss (for logging)
        if fg_mask.sum():
            # Filter the loss tensor to only positive samples
            cls_losses_pos = cls_losses[fg_mask] # [N_positive, N_classes]
            
            for i in range(self.nc):
                # Mask for positive samples belonging to true class 'i'
                class_mask = (gt_classes == i)
                if class_mask.sum() > 0:
                    # Sum the loss only for the true class index 'i'
                    class_loss_i = cls_losses_pos[class_mask, i].sum()
                    self.cls_loss_log[i] += class_loss_i.detach()
                    self.cls_counts_log[i] += class_mask.sum().detach()

        # Original Aggregation (for backprop)
        # The total CLS loss for backprop is the sum of losses *only for the positive class* # for all positive samples (normalized later).
        if fg_mask.sum():
             # Get the loss only for the assigned (true) class for each positive sample
            lcls_total = cls_losses_pos[torch.arange(cls_losses_pos.shape[0]), gt_classes].sum()
            loss[1] = lcls_total / target_scores_sum # This is the main aggregated cls loss
        else:
             loss[1] = torch.zeros(1, device=self.device)
            
        # Bbox and DFL Loss (Rest of the original __call__)
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        
        # Final Return and Logging Cleanup

        # Normalize the class-wise loss for logging (mean loss per sample for that class)
        non_zero_counts = torch.where(self.cls_counts_log > 0, self.cls_counts_log, torch.ones_like(self.cls_counts_log))
        mean_cls_loss_per_class = (self.cls_loss_log / non_zero_counts).tolist()

        # Create a list of the 3 main loss components (detached)
        main_loss_items = loss.detach().tolist()
        
        # Add the class-wise loss items to the logging list
        # This list of metrics is what the Trainer will log
        loss_items = main_loss_items + mean_cls_loss_per_class

        # Reset accumulators for the next batch
        self.cls_loss_log = torch.zeros(self.nc, device=self.device)
        self.cls_counts_log = torch.zeros(self.nc, device=self.device)

        # Debug print for final loss values
        if RANK in {-1, 0} and (self._call_count == 1 or self._call_count % 100 == 0):
            print(f"[CustomDetectionLoss.__call__] Final losses - box: {loss[0].item():.4f}, cls: {loss[1].item():.4f}, dfl: {loss[2].item():.4f}, total: {loss.sum().item() * batch_size:.4f}", flush=True)
            sys.stdout.flush()

        # Return the scaled loss for backprop and the loss items for logging
        return loss.sum() * batch_size, torch.tensor(loss_items, device=self.device)

class CustomWeightedTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initializes the CustomWeightedTrainer and loads the model object.
        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict): Overrides for the configuration.
            _callbacks (dict): Callbacks for training events.
        """
        super().__init__(cfg, overrides, _callbacks)
        print("✅ CustomWeightedTrainer Initialized.")
        if isinstance(self.model, str):
            # only load if model is a string path
            try:
                loaded_model_instance = YOLO(self.model)
                self.model = loaded_model_instance.model
                self.stride = loaded_model_instance.stride  # Also set the stride property for safety
            except Exception as e:
                print(f"Warning: Failed to pre-load model object using YOLO constructor: {e}")
        # Replace the default loss function with the custom one
        

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        Overrides the base method to inject WeightedRandomSampler for training.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."

        # This section is directly copied/adapted from the base DetectionTrainer
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size) # This loads data
            
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            shuffle = False
        # Inserting the weighted sampler for training mode
        if mode == 'train':
            # Calculate weights using your function
            weights = make_image_weights(dataset, alpha=ALPHA_FOR_SAMPLER)
            
            # Create the custom sampler
            sampler = WeightedRandomSampler(
                weights, num_samples=len(dataset), replacement=True
            )
            
            # Use the PyTorch DataLoader to build the sampler into the loader
            base_loader = InfiniteDataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,               # Inject the sampler here
                num_workers=self.args.workers,
                pin_memory=True,
                collate_fn=getattr(dataset, 'collate_fn', None), # Get collate_fn from dataset or default
                drop_last=False
            )
            
            # The returned loader MUST be converted back to the InfiniteDataLoader
            return base_loader

        # For validation, we use the standard build_dataloader (without the custom sampler)
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def train(self):
        print("✅ CustomWeightedTrainer train Called.")
        """Overrides DetectionTrainer.train to ensure custom loss is used."""
        return super().train()  # Call the parent method which handles the training process
    
    def _do_train(self):
        print("✅ CustomWeightedTrainer _do_train Called.")
        """Overrides DetectionTrainer._do_train to ensure custom loss is used."""
        super()._do_train()  # Call the parent method which handles the training loop
    # changing the setuptrain method to add the custom loss tracking
    def _setup_train(self):
        print("✅ CustomWeightedTrainer _setup_train Called.")
        """Overrides the parent BaseTrainer._setup_train (via DetectionTrainer's inheritance) 
        to inject custom loss item names after initialization."""
        
        # 1. Call parent's setup_train method (actually runs BaseTrainer._setup_train)
        # This method handles model compilation, freezing, DDP setup, dataloader creation,
        # and most importantly, sets self.loss_names to ('box_loss', 'cls_loss', 'dfl_loss').
        super()._setup_train() 
        global CLASS_WEIGHTS
        if CLASS_WEIGHTS is not None:
            print("✅ Using Class Weights in CustomWeightedTrainer.")
            train_dataset = self.train_loader.dataset 
            cls_counts = get_class_counts(train_dataset)
            CLASS_WEIGHTS = calculate_inverse_frequency_weights(cls_counts)
        self.loss = CustomDetectionLoss(self.model, class_weights=CLASS_WEIGHTS)
        
        # 2. Add the custom class-wise loss names
        nc = self.loss.nc # Get the corrected class count (9) from your custom loss module
        
        custom_loss_names = [f'cls_loss_C{i}' for i in range(nc)]
        print("Custom class-wise loss names to add:", custom_loss_names)
        
        new_loss_names = list(self.loss_names) # Start with ['box_loss', 'cls_loss', 'dfl_loss']
        
        # Insert or append the 9 new names
        if new_loss_names[-1] == 'dfl_loss':
            # Append 9 custom loss names after the 3 standard ones
            new_loss_names.extend(custom_loss_names)
        
        self.loss_names = new_loss_names
        
        # Update the validator's loss names too (used in final evaluation log headers)
        if self.validator:
            self.validator.loss_names = self.loss_names

        print(f"✅ Trainer Logging: Updated loss_names to include {nc} custom class-wise losses.")


    # def label_loss_items(self, loss_items=None, prefix="train"):
    #     """
    #     Returns a dictionary of loss metrics (or list of names if loss_items is None).
    #     Handles both 3-item (val) and 12-item (train) losses.
    #     """
    #     # 1. Handle Header Names (loss_items is None)
    #     if loss_items is None:
    #         # self.loss_names (12 items) is correct here.
    #         return self.loss_names 
        
    #     # 2. Handle Loss Values (loss_items is the tensor/tuple of values)
        
    #     # Convert to a list of Python floats regardless of original type (tensor/tuple)
    #     if isinstance(loss_items, torch.Tensor):
    #         loss_values = loss_items.tolist()
    #     elif isinstance(loss_items, (list, tuple)):
    #         # Crucial: Convert any nested tensors to floats too, and flatten if needed
    #         # For validation, it often passes a 3-item tuple of tensors, not your custom 12.
    #         # We must handle the case where it's 3 items (box, cls, dfl) from the Validator.
            
    #         flat_loss_values = []
    #         for item in loss_items:
    #             if isinstance(item, torch.Tensor):
    #                 flat_loss_values.extend(item.tolist())
    #             elif isinstance(item, (float, int)):
    #                 flat_loss_values.append(item)
    #             else:
    #                 # Catch the case where an unexpected tuple/list might be passed
    #                 # This is likely where your original code failed to flatten/convert
    #                 flat_loss_values.extend(list(item))
    #         loss_values = flat_loss_values
    #     else:
    #         loss_values = list(loss_items) # Fallback

    #     # Check if it's the 3-item validation output or the 12-item training output
    #     if len(loss_values) == 3 and len(self.loss_names) == 12:
    #         # This is validation output (only box, cls, dfl are returned by the Validator)
    #         # Ultralytics Validator does not calculate the class-wise losses.
    #         # Pad the remaining 9 custom losses with 0.0 for consistent logging structure
    #         # The base 3 loss names must match the first 3 values in self.loss_names
    #         loss_values.extend([0.0] * 9) 

    #     # Final check for size consistency
    #     if len(loss_values) != len(self.loss_names):
    #          raise ValueError(f"Loss length mismatch: Expected {len(self.loss_names)}, got {len(loss_values)}")

    #     # Map the 12 loss values to the 12 loss names (headers)
    #     all_losses_dict = dict(zip(self.loss_names, loss_values))

    #     # The BaseTrainer handles the 'train/' or 'val/' prefix for the keys
    #     return all_losses_dict

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
        }
        if augmentations:
            custom_overrides.update(augmentations)

        model.overrides.update(custom_overrides)
        model.trainer = CustomWeightedTrainer(overrides=model.overrides)
        for i in callbacks:
            model.add_callback(i, callbacks[i])
        results = model.trainer.train()
        
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
def get_classwise_results(model_path: str, yaml_path: str = "data.yaml"):
    """
    This function loads a trained YOLO model and retrieves class-wise precision results and prints them.
    Args:
        model_path (str): Path to the trained YOLO model file (e.g., best.pt).
    Returns:
        None
    """
    model = YOLO(model_path)
    results = model.val(data = yaml_path)
    # The class names are stored in the model object
    class_names = results.names 

    # Access the list of dictionaries containing per-class metrics
    class_metrics = results.box.class_result

    
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
    # print("Starting the training script...")
    # train model based on where the data is
    # model = train_with_data_in_cloud()
    model = train_with_data_locally("2025-10-09_len_14200")
    #path = r"ultralytics_yolo_projectAug_overloadeed_yolov11l/yolov11n_run/weights/best.pt"
    #yaml_path = r"2025-10-09_len_14200/data.yaml"
    #get_classwise_results(path, yaml_path)

