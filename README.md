# CityVision

## Overview

CityVision is a tool that allows the use of traffic videos to extract information about the traffic flow in a city. The tool is able to detect vehicles, track them, and count the number of vehicles that pass through defined zones. The tool will output data in tabular format (Pandas DataFrame) and can generate annotated videos.

## Features

*   **Vehicle Detection:** Utilizes YOLO models to detect various vehicle types from video frames.
*   **Vehicle Tracking:** Employs tracking algorithms (e.g., ByteTrack) to follow detected vehicles across frames.
*   **Vehicle Counting:** Counts vehicles that cross predefined polygonal zones in the video.
*   **Customizable Configuration:** Allows configuration of model parameters, detection classes, tracker settings, and counting zones.
*   **Reporting:** Generates reports in Pandas DataFrame format and can save annotated videos.
*   **Google Cloud Storage Integration:** Includes functionality to save reports and videos to GCS.

## Directory Structure

*   `cityvision/`: Main library source code.
    *   `cityvision/models/main.py`: Contains the core `yolo_counting_model` class for detection, tracking, and counting.
*   `configs/`: Configuration files.
    *   `bytetrack.yml`: Configuration for the ByteTrack tracker.
    *   `cityvision.yml`: Configuration for YOLO model class names and numeric IDs.
*   `main.py`: Entry point for a launcher script (appears to be related to Google Cloud Dataflow).
*   `requirements.txt`: List of Python dependencies for the project.
*   `setup.py`: Standard Python setup script for packaging and distribution.
*   `Dockerfile`: Defines a Docker image, primarily for deploying as a Google Cloud Dataflow Flex Template.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `LICENSE`: Contains the GNU Affero General Public License v3.0 under which the project is released.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd cityvision
    ```
2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```
3.  The project also includes a `setup.py` file, which can be used for installation if desired:
    ```bash
    pip install .
    ```
    Note: This project has specific dependencies like `torch` with CUDA, which might require careful installation depending on your system setup. Refer to the `Dockerfile` for more detailed environment setup if you encounter issues.

## Configuration

The behavior of the `yolo_counting_model` is controlled by a configuration dictionary passed during its initialization and by YAML configuration files.

*   **`configs/cityvision.yml`**: This file defines the mapping between class names and their corresponding numerical IDs used by the YOLO model.
    ```yaml
    # Example from configs/cityvision.yml
    nc: 11
    names: ['articulated_truck', 'bicycle', 'bus', 'car', 'motorcycle',
               'motorized_vehicle', 'non-motorized_vehicle', 'pedestrian',
               'pickup_truck', 'single_unit_truck', 'work_van']
    ```
*   **`configs/bytetrack.yml`**: This file contains settings for the ByteTrack object tracker, such as tracking thresholds and buffer sizes.
    ```yaml
    # Example from configs/bytetrack.yml
    tracker_type: bytetrack
    track_high_thresh: 0.3
    track_low_thresh: 0.1
    # ... other tracker settings
    ```
*   **`yolo_counting_model` Configuration (`__init__` parameters):**
    The main class `yolo_counting_model` in `cityvision/models/main.py` is initialized with a configuration dictionary that includes:
    *   `name` (str): The name of the YOLO model file to use (e.g., `yolov8n.pt`).
    *   `config` (dict): A dictionary containing various settings:
        *   `study_name` (str): A name for the study or analysis session.
        *   `iou_threshold` (float): Intersection over Union threshold for NMS.
        *   `confidence_threshold` (float): Minimum detection confidence.
        *   `polygons` (dict): A dictionary where keys are direction names (e.g., "EB", "WB") and values are lists of points defining polygonal zones for counting.
        *   `classes` (dict): A dictionary mapping class IDs (int) to class names (str), which should align with `configs/cityvision.yml`.
        *   `tracker_config` (str): Path to the tracker configuration file (e.g., `configs/bytetrack.yml`).
        *   `report_path` (str): Path to save reports and output videos.
        *   `direction` (list): A list of two strings representing the directions corresponding to the polygon keys (e.g., `["EB", "WB"]`).

## How to Run

While `main.py` serves as an entry point for a Dataflow launcher, you can use the `yolo_counting_model` class directly for video processing.

Here's a basic example of how to import and use the `yolo_counting_model`:

```python
from cityvision.models.main import yolo_counting_model
import yaml # For loading class names if needed, or define manually

# --- Configuration ---
# 1. Load class names (example - adapt as needed)
# This would typically come from your cityvision.yml or be defined according to your model
# For this example, let's manually define a subset based on the example cityvision.yml
# Ensure these class IDs (0, 1, 2 etc.) match what your YOLO model was trained on.
active_classes = {
    3: "car",       # Assuming 'car' is class ID 3 in your cityvision.yml
    2: "bus"        # Assuming 'bus' is class ID 2 in your cityvision.yml
    # Add other classes your model detects and you want to count
}

# 2. Define polygons for counting (coordinates depend on your video resolution and scene)
# Each polygon is a list of [x, y] points.
polygons_config = {
    "EastBound": [[100, 200], [300, 200], [300, 250], [100, 250]], # Example polygon for Eastbound traffic
    "WestBound": [[400, 300], [600, 300], [600, 350], [400, 350]]  # Example polygon for Westbound traffic
}

# 3. Model and run configuration
model_config = {
    "study_name": "my_traffic_analysis",
    "iou_threshold": 0.45,              # IoU threshold for Non-Maximum Suppression
    "confidence_threshold": 0.25,       # Minimum detection confidence
    "polygons": polygons_config,
    "classes": active_classes,
    "tracker_config": "configs/bytetrack.yml", # Path to tracker settings file
    "report_path": "./output_reports/",        # Directory to save outputs
    "direction": ["EastBound", "WestBound"]    # Must match keys in polygons_config
}

# --- Initialization ---
# Initialize the model. Replace 'yolov8n.pt' with your actual YOLO model file.
# The model file should be accessible at the provided path or name.
try:
    # Ensure you have a YOLO model file (e.g., yolov8n.pt) downloaded or trained.
    # You might need to place it in a location where the script can find it.
    traffic_model = yolo_counting_model(name='yolov8n.pt', config=model_config)
    print("Model initialized successfully.")
except Exception as e:
    print(f"Error initializing model: {e}")
    print("Please ensure 'yolov8n.pt' (or your specified model) is accessible and all configurations are correct.")
    exit()

# --- Running Analysis ---
# Provide the path to your video file and a descriptive name.
video_file_path = "path/to/your/traffic_video.mp4" # IMPORTANT: Replace with your video path
video_file_name = "traffic_study_siteA_20230101_1000.mp4" # Example file name

# Before running, ensure the video_file_path exists.
# For a quick test, you might need a sample video.
# Example (commented out to prevent errors if path is invalid):
#
# import os
# if os.path.exists(video_file_path):
#     print(f"Starting analysis for {video_file_path}...")
#     traffic_model.run(file_path=video_file_path, file_name=video_file_name)
#     print(f"Analysis finished. Outputs should be in {model_config['report_path']}")
#
#     # --- Generating Report ---
#     # Generate a summary report (after running the analysis)
#     # The UUID can be any unique identifier for this run.
#     report_df = traffic_model.generate_report(uuid="traffic_analysis_run_001")
#     print("\nGenerated Report:")
#     print(report_df)
# else:
#     print(f"Video file not found: {video_file_path}")
#     print("Please replace 'path/to/your/traffic_video.mp4' with an actual video file path to run the analysis.")

print("\nREADME Example: Setup complete. Uncomment and adjust paths to run analysis.")

```

**Note:** The Python code example above is for illustration. You will need to:
1.  Replace `'yolov8n.pt'` with the actual YOLO model file you intend to use (e.g., `yolov8s.pt`, `yolov8m.pt`, or a custom trained model). Make sure this model file is available.
2.  Adjust `active_classes` to reflect the classes your model can detect and that you are interested in counting. The class IDs must match those used during your YOLO model's training and defined in `configs/cityvision.yml`.
3.  Modify `polygons_config` with coordinates appropriate for your video's resolution and the areas where you want to count vehicles.
4.  Change `video_file_path` to the actual path of the video you want to process.
5.  The `cityvision/models/main.py` script attempts to use a CUDA-enabled GPU by default. If a GPU is not available or not set up correctly with PyTorch, it may fall back to ONNX export/usage or CPU, which can be slower.

## Output

The `yolo_counting_model` generates the following outputs:

*   **Annotated Videos:** Saves videos in the `report_path/video/` directory, showing detected bounding boxes, tracking IDs, and counts of objects crossing the defined polygons.
*   **Tabular Data:** The `generate_report()` method returns a Pandas DataFrame containing the counts of objects, resampled at specified intervals, for each direction. This DataFrame includes columns for timestamps, class counts, direction, and a UUID.
*   **Low Confidence Detections (Optional):** If detections fall below a certain threshold, images might be saved to `report_path/low_confidence_detections/` for review.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See the `LICENSE` file for full details.
