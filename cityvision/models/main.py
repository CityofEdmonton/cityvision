import pathlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import cv2
from PIL import Image
import yaml
import os
import time
import supervision as sv
import pandas as pd
from tqdm import tqdm
import logging
import os
import torch
import google.cloud.storage as storage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def get_depth_map(frame: np.ndarray, camera_intrinsic: os.path) -> np.ndarray:
    """
    Computes a depth map from a video frame using the camera intrinsic parameters.

    Args:
        frame: The video frame (as a NumPy array) to process.
        camera_intrinsic: Camera intrinsic parameters file path

    Returns:
        A depth map (as a NumPy array) representing the depth of each pixel in the frame.
    """

    # read camera intrinsic parameters from yml file
    with open(camera_intrinsic, "r") as file:
        camera_intrinsic = yaml.safe_load(file)

    if "dis_vec" not in camera_intrinsic:
        raise ValueError("Camera intrinsic parameters must include 'dis_vec'.")
    if not isinstance(frame, np.ndarray):
        raise TypeError("Frame must be a NumPy array.")
    if frame.ndim != 3 or frame.shape[2] not in [3, 4]:
        raise ValueError("Frame must be a 3-channel (RGB) or 4-channel (RGBA) image.")
    if not isinstance(camera_intrinsic, dict):
        raise TypeError("Camera intrinsic parameters must be a dictionary.")
    if not all(key in camera_intrinsic for key in ["fx", "fy", "cx", "cy", "dis_vec"]):
        raise ValueError(
            "Camera intrinsic parameters must include 'fx', 'fy', 'cx', 'cy', and 'dis_vec'."
        )

    h, w = frame.shape[:2]
    print("camera distortion vector", camera_intrinsic["dis_vec"])

    # Create camera matrix
    camera_matrix = np.array(
        [
            [camera_intrinsic["fx"], 0, camera_intrinsic["cx"]],
            [0, camera_intrinsic["fy"], camera_intrinsic["cy"]],
            [0, 0, 1],
        ]
    )

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, np.array(camera_intrinsic["dis_vec"]), (w, h), 1, (w, h)
    )

    # undistort
    undistorted_img = cv2.undistort(
        frame, camera_matrix, np.array(camera_intrinsic["dis_vec"]), None, newcameramtx
    )

    # crop the image
    x, y, w, h = roi
    undistorted_img = undistorted_img[y : y + h, x : x + w]

    undistorted_image = Image.fromarray(undistorted_img)

    CHECKPOINT = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"

    image_processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
    model = AutoModelForDepthEstimation.from_pretrained(CHECKPOINT)

    inputs = image_processor(images=undistorted_image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = (
        torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=undistorted_image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return (
        prediction,
        camera_matrix,
        np.array(camera_intrinsic["dis_vec"]),
        roi,
        newcameramtx,
    )


class yolo_counting_model:
    """
    Manages vehicle detection, tracking, and counting using YOLO models.

    This class encapsulates the logic for processing video files, detecting objects
    (typically vehicles) using a YOLO model, tracking these objects across frames,
    and counting them as they cross predefined polygonal zones. It also handles
    report generation and saving results, including annotated videos and data,
    potentially to Google Cloud Storage.
    """

    def __init__(self, frame: np.ndarray, name: str, config: dict) -> None:
        """
        Initializes the yolo_counting_model.

        Args:
                name: The name or path of the YOLO model file (e.g., 'yolov8n.pt').
                config: A dictionary containing configuration parameters:
                        study_name (str): Identifier for the current analysis study.
                        iou_threshold (float): Intersection over Union threshold for NMS.
                        confidence_threshold (float): Minimum detection confidence.
                        polygons (dict): Defines counting zones. Keys are zone names (e.g., "EB", "WB"),
                                                         values are lists of [x, y] points defining the polygon.
                        classes (dict): Mapping of class IDs (int) to class names (str)
                                                        (e.g., {0: 'car', 1: 'truck'}).
                        tracker_config (str): Path to the tracker configuration file (e.g., 'bytetrack.yml').
                        report_path (str): Directory path to save output reports and videos.
                        direction (list): List of two strings representing the primary directions
                                                          of movement corresponding to polygon keys.
        """
        self.model_name = name
        self.first_frame = frame
        (
            self.depth_map,
            self.camera_intrinsic,
            self.camera_distortion,
            self.roi,
            self.new_camera_matrix,
        ) = get_depth_map(frame, config["camera_intrinsic"])
        self.study_name = config["study_name"]
        self.iou_threshold = config["iou_threshold"]
        self.confidence_threshold = config["confidence_threshold"]
        self.direction_vector = config[
            "direction_vector"
        ]  # Dict of vector directions e.g. {"EB": [[x1, y1], [x2, y2]], "WB": [[x1, y1], [x2, y2]]}
        self.classes = config["classes"]
        self.tracker_config = config["tracker_config"]
        self.report_path = config["report_path"]
        self.direction = config["direction"]

        # create dictionary to store crossed objects according to key values
        self.crossed_objects = {self.direction[0]: {}, self.direction[1]: {}}
        self.track_history = defaultdict(
            lambda: {"track": [], "speed": []}
        )  # format {track_id: {"track": [], "speed": []}}
        self.count = 0
        self.model = YOLO(self.model_name)

        try:
            self.model.to("cuda")
        except:
            self.model.export(format="onnx", half=False, imgsz=(384, 576))
            self.model = YOLO(self.model_name.split(".")[0] + ".onnx")

        if "WB" in self.direction_vector.keys():
            self.dir_1 = self.direction_vector["EB"]
            self.dir_2 = self.direction_vector["WB"]

        else:
            self.dir_1 = self.direction_vector["NB"]
            self.dir_2 = self.direction_vector["SB"]

        self.classes_ids = [k for k, _ in self.classes.items()]

    def reset(self) -> None:
        """
        Resets the internal state of the counter.

        This includes clearing the history of crossed objects, track history,
        and the frame count. Useful for processing multiple videos sequentially
        with the same model instance.
        """
        self.crossed_objects = {self.direction[0]: {}, self.direction[1]: {}}
        self.track_history = defaultdict(lambda: {"track": [], "speed": []})
        self.count = 0

    def run(self, file_path: str, file_name: str) -> None:
        """
        Processes a video file for object detection, tracking, and counting.

        Reads a video frame by frame, applies YOLO detection and tracking,
        counts objects crossing defined polygons, and saves an annotated video.

        Args:
                file_path: The full path to the video file.
                file_name: The name of the video file (used for naming outputs
                                   and extracting metadata like start time).
        """

        start_time = datetime.strptime(file_name.split("_")[1], "%Y%m%d%H%M")

        cap = cv2.VideoCapture(file_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )

        logging.info("frame_width: " + str(w))
        logging.info("frame_height: " + str(h))
        logging.info("fps: " + str(fps))

        frame_generator = sv.get_video_frames_generator(source_path=file_path)

        # Open a video sink for the output video
        video_info = sv.VideoInfo.from_video_path(file_path)
        if not pathlib.Path(self.report_path + "video/").exists():
            pathlib.Path(self.report_path + "video/").mkdir(parents=True, exist_ok=True)

        video_report_path = (
            self.report_path + "video/" + file_name.split(".")[0] + ".mp4"
        )
        with sv.VideoSink(video_report_path, video_info) as sink:
            for frame in tqdm(frame_generator, total=video_info.total_frames):
                success, frame = cap.read()
                self.count += 1
                if success and self.count % 1 == 0:
                    annotated_frame = self.get_count(
                        frame, start_time, file_name.split(".")[0]
                    )

                    # Draw the arrow line on the frame
                    cv2.arrowedLine(
                        annotated_frame,
                        tuple(self.dir_1[0]),
                        tuple(self.dir_1[1]),
                        (0, 0, 0),
                        2,
                    )
                    cv2.arrowedLine(
                        annotated_frame,
                        tuple(self.dir_2[0]),
                        tuple(self.dir_2[1]),
                        (0, 0, 0),
                        2,
                    )

                    # Write the count of objects on each frame
                    count_text_1 = f"Objects crossed {self.direction[0]}: {len(self.crossed_objects[self.direction[0]])}"
                    count_text_2 = f"Objects crossed {self.direction[1]}: {len(self.crossed_objects[self.direction[1]])}"

                    cv2.putText(
                        annotated_frame,
                        count_text_1,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        annotated_frame,
                        count_text_2,
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        2,
                    )

                    # Write the frame with annotations to the output video
                    sink.write_frame(annotated_frame)
                else:
                    pass

        # Release the video capture
        cap.release()
        print(f"Data has been written to {self.report_path}")
        print(count_text_1)
        print(count_text_2)

    def get_count(
        self, frame: np.ndarray, start_time: datetime, video_name: str
    ) -> np.ndarray:
        """
        Processes a single frame to detect, track, and count objects.

        Identifies objects crossing predefined polygons within the frame and
        updates the count. Annotates the frame with bounding boxes, tracks,
        and crossing information.

        Args:
                frame: The video frame (as a NumPy array) to process.
                start_time: The starting datetime of the video recording, used for
                                        timestamping events.
                video_name: The name of the video (without extension), used for
                                        saving auxiliary data like low-confidence detections.

        Returns:
                The annotated frame (as a NumPy array) with visualizations.
        """

        results, boxes, class_ids, class_names, annotated_frame = self.get_result(
            frame, video_name
        )

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.cpu().int().tolist()

            # Plot the tracks and count objects crossing the line
            for box, track_id, cls in zip(boxes, track_ids, class_names):
                x, y, w, h = box
                pt = (int(x.numpy()), int(y.numpy()))
                cls = cls
                track = self.track_history[track_id].get("track", [])
                speed = self.track_history[track_id].get("speed", [])
                track.append((float(x), float(y)))  # x, y center point

                # annotate the bounding box
                cv2.rectangle(
                    annotated_frame,
                    (int(x - w / 2), int(y - h / 2)),
                    (int(x + w / 2), int(y + h / 2)),
                    (255, 0, 0),
                    2,
                )
                if len(track) > 60:  # retain 30 tracks for 30 frames
                    track.pop(0)

                speed_estimate = "N/A"
                if len(track) > 15:
                    # calculate speed and direction if more than 15 points
                    speed_estimate = self.calculate_speed(track, annotated_frame)
                    if speed_estimate != "N/A":
                        speed.append(speed_estimate)
                        speed_estimate = np.mean(
                            speed
                        )  # average speed over last 15 frames

                    # get direction vector and compare with the polygon direction

                    direction_vector_1 = np.array(self.dir_1[1]) - np.array(
                        self.dir_1[0]
                    )
                    direction_vector_2 = np.array(self.dir_2[1]) - np.array(
                        self.dir_2[0]
                    )

                    vehicle_vector = np.array(track[-1]) - np.array(track[0])
                    vehicle_vector = vehicle_vector / np.linalg.norm(vehicle_vector)

                    # Check if the vehicle is moving in the direction of the polygon
                    direction_angle_1 = np.arccos(
                        np.clip(
                            np.dot(
                                vehicle_vector,
                                direction_vector_1 / np.linalg.norm(direction_vector_1),
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    direction_angle_2 = np.arccos(
                        np.clip(
                            np.dot(
                                vehicle_vector,
                                direction_vector_2 / np.linalg.norm(direction_vector_2),
                            ),
                            -1.0,
                            1.0,
                        )
                    )

                    logging.info(
                        f"Direction angle 1 for {track_id}: {direction_angle_1}"
                    )
                    logging.info(
                        f"Direction angle 2 for {track_id}: {direction_angle_2}"
                    )

                    if direction_angle_1 < np.pi / 2:  # 90 degrees
                        # Object is moving in the direction of dir_1
                        if track_id not in self.crossed_objects[self.direction[0]]:
                            time_seen = datetime.fromtimestamp(
                                int(self.count / 30) + start_time.timestamp()
                            )
                            self.crossed_objects[self.direction[0]][track_id] = [
                                time_seen.strftime("%Y-%m-%d %H:%M:%S"),
                                cls,
                            ]

                    else:
                        # Object is moving in the direction of dir_2
                        if track_id not in self.crossed_objects[self.direction[1]]:
                            time_seen = datetime.fromtimestamp(
                                int(self.count / 30) + start_time.timestamp()
                            )
                            self.crossed_objects[self.direction[1]][track_id] = [
                                time_seen.strftime("%Y-%m-%d %H:%M:%S"),
                                cls,
                            ]

                # Create formatted text with better layout and colors
                speed_text = (
                    f"{speed_estimate:.1f} km/h" if speed_estimate != "N/A" else "N/A"
                )
                direction_text = ""
                if track_id in self.crossed_objects[self.direction[0]]:
                    direction_text = self.direction[0]
                elif track_id in self.crossed_objects[self.direction[1]]:
                    direction_text = self.direction[1]
                else:
                    direction_text = "Unknown"

                # Main label with ID and class
                cv2.putText(
                    annotated_frame,
                    f"ID: {track_id} | {cls}",
                    (int(x - w / 2), int(y - h / 2) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 0),
                    2,
                )

                # Direction label
                cv2.putText(
                    annotated_frame,
                    f"Direction: {direction_text}",
                    (int(x - w / 2), int(y - h / 2) + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                # Speed label
                cv2.putText(
                    annotated_frame,
                    f"Speed: {speed_text}",
                    (int(x - w / 2), int(y - h / 2) + 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2,
                )

                # Annotate center of the object
                cv2.circle(annotated_frame, pt, 5, (0, 255, 0), -1)

        return annotated_frame

    def get_result(self, frame: np.ndarray, video_name: str) -> tuple:
        """
        Performs object detection and tracking on a single frame.

        Uses the configured YOLO model to detect objects and applies tracking.
        Optionally saves images of low-confidence detections.

        Args:
                frame: The video frame (as a NumPy array) to process.
                video_name: The name of the video (without extension), used if
                                        saving low-confidence detection images.

        Returns:
                A tuple containing:
                        - results: Raw results from the YOLO model's track method.
                        - boxes: Detected bounding boxes (xywh format).
                        - class_ids: List of class IDs for detected objects.
                        - class_names: List of class names for detected objects.
                        - annotated_frame: The frame annotated with detections by the model.
        """

        results = self.model.track(
            frame,
            classes=self.classes_ids,
            persist=True,
            save=False,
            tracker=self.tracker_config,
            imgsz=(384, 576),
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            agnostic_nms=False,
        )

        # save low confidence detections
        conf_list = results[0].boxes.conf.cpu().float().tolist()
        if any(conf_el < 0.1 for conf_el in conf_list):
            with sv.ImageSink(
                target_dir_path=self.report_path
                + "low_confidence_detections/"
                + video_name,
                overwrite=False,
            ) as sink:
                sink.save_image(image=frame, image_name=f"image_{self.count}.jpg")

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        class_ids = results[0].boxes.cls.cpu().int().tolist()

        class_names = [self.classes[i] for i in class_ids]

        # Visualize the results on the frame
        # annotated_frame = results[0].plot()
        annotated_frame = frame.copy()

        return (results, boxes, class_ids, class_names, annotated_frame)

    def resample_data(
        self, df_first: pd.DataFrame, interval: str = "1min"
    ) -> pd.DataFrame:
        """
        Resamples time-series data of detected objects to a specified interval.

        Args:
                df_first: Pandas DataFrame with a 'timestamp' column and object data.
                                  Typically contains one row per detected object crossing.
                interval: Pandas resampling interval string (e.g., '1min', '15min', '1H').
                                  Defaults to '1min'.

        Returns:
                A new Pandas DataFrame with data aggregated by the specified interval.
        """

        df = df_first.copy()
        # Reset the index to make the track_id a column
        df.reset_index(inplace=True)
        # Set the index to the timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)
        df.rename(columns={"index": "Track ID"}, inplace=True)
        # Resample the data by intervals
        resampled_df = df.resample(interval).count()

        # Reset the index to make the time intervals a column
        resampled_df = resampled_df.reset_index()

        # Rename the time interval column to 'timestep'
        resampled_df.rename(columns={"index": "timestep"}, inplace=True)

        return resampled_df.copy()

    def _undistort_pt(self, pt_matrix: list, frame: np.ndarray) -> np.ndarray:

        h, w = frame.shape[:2]

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_intrinsic, self.camera_distortion, (w, h), 1, (w, h)
        )
        undst_pt = cv2.undistortPoints(
            pt_matrix,
            self.camera_intrinsic,
            self.camera_distortion,
            None,
            P=newcameramtx,
        )

        undst_pt = undst_pt.squeeze()
        x, y, w, h = roi
        undst_pt[0] -= x
        undst_pt[1] -= y

        return undst_pt

    def calculate_distance(self, x1, y1, x2, y2):
        # Get the depth values at the two points

        if (
            y1 < 0
            or y1 >= self.depth_map.shape[0]
            or x1 < 0
            or x1 >= self.depth_map.shape[1]
        ):
            return "N/A"

        if (
            y2 < 0
            or y2 >= self.depth_map.shape[0]
            or x2 < 0
            or x2 >= self.depth_map.shape[1]
        ):
            return "N/A"

        z1 = self.depth_map[y1, x1]
        z2 = self.depth_map[y2, x2]

        camera_matrix = self.camera_intrinsic

        # Calculate the 3D coordinates of the two points
        p1 = np.array(
            [
                (x1 - camera_matrix[0, 2]) * z1 / camera_matrix[0, 0],
                (y1 - camera_matrix[1, 2]) * z1 / camera_matrix[1, 1],
                z1,
            ]
        )
        p2 = np.array(
            [
                (x2 - camera_matrix[0, 2]) * z2 / camera_matrix[0, 0],
                (y2 - camera_matrix[1, 2]) * z2 / camera_matrix[1, 1],
                z2,
            ]
        )

        # Calculate the distance between the two points
        distance = np.linalg.norm(p2 - p1) / 1.5

        return distance

    def calculate_speed(self, track: list, frame: np.ndarray):
        """
        Calculates the speed of a tracked object based on its trajectory.

        Args:
                track: A list of tuples representing the (x, y) coordinates of the object's trajectory.
                fps: Frames per second of the video, used to convert pixel distance to real-world speed.

        Returns:
                The calculated speed in pixels per second.
        """

        undst_pt_1 = self._undistort_pt(np.array(track[-1]), frame)
        undst_pt_2 = self._undistort_pt(np.array(track[-15]), frame)
        x1, y1 = undst_pt_1[0], undst_pt_1[1]
        x2, y2 = undst_pt_2[0], undst_pt_2[1]

        distance = self.calculate_distance(
            int(x2),
            int(y2),
            int(x1),
            int(y1),
        )

        if distance != "N/A":
            time = 15 / 30
            speed = distance / time * 3.6
            return speed
        else:
            return "N/A"

    def generate_report(self, uuid: str) -> pd.DataFrame:
        """
        Generates a consolidated report of object counts for all directions.

        The report is a Pandas DataFrame with counts aggregated by time intervals
        (using `resample_data`) for each direction.

        Args:
                uuid: A unique identifier for this analysis run or report.

        Returns:
                A Pandas DataFrame containing the traffic count report, with columns
                for timestamp, class counts, direction, and the provided UUID.
        """

        print(self.crossed_objects)

        df_1 = pd.DataFrame.from_dict(
            self.crossed_objects[self.direction[0]],
            orient="index",
            columns=["timestamp", "Class"],
        )
        df_2 = pd.DataFrame.from_dict(
            self.crossed_objects[self.direction[1]],
            orient="index",
            columns=["timestamp", "Class"],
        )

        df_1 = self.resample_data(df_1)
        df_2 = self.resample_data(df_2)

        df_1["Direction"] = self.direction[0]
        df_2["Direction"] = self.direction[1]

        resampled_df = pd.concat([df_1, df_2])
        resampled_df["uuid"] = uuid

        print("df", resampled_df)

        return resampled_df

    def save_to_gcs(
        self,
        storage_client: storage.Client,
        bucket_name: str,
        uuid: str,
        video_name: str,
    ) -> None:
        """
        Saves generated reports and annotated videos to Google Cloud Storage.

        Uploads the annotated video and potentially low-confidence detection images
        to specified GCS buckets and paths.

        Args:
                storage_client: An initialized Google Cloud Storage client instance.
                bucket_name: The name of the GCS bucket for general video outputs.
                                         (Note: A hardcoded bucket "processed-cityvision" is also used).
                uuid: A unique identifier, used in structuring the GCS path for images.
                video_name: The name of the video file (used for naming blobs in GCS).
        """
        # save report folder to gcs
        bucket_videos = storage_client.bucket(
            bucket_name
        )  # Corrected to use bucket_name
        bucket_training_images = storage_client.bucket("processed-cityvision")
        video_name = video_name.split("/")[-1]
        # upload folder to bucket

        blob_video = bucket_training_images.blob(
            self.study_name + "/annotated_videos/" + video_name
        )
        blob_video.upload_from_filename(self.report_path + "video/" + video_name)

        self.move_folder_gcs(
            bucket_videos, self.study_name, bucket_training_images, self.study_name
        )

        if pathlib.Path(
            self.report_path + "low_confidence_detections/" + video_name.split(".")[0]
        ).exists():
            for file in os.listdir(
                self.report_path
                + "low_confidence_detections/"
                + video_name.split(".")[0]
            ):
                blob_images = bucket_training_images.blob(
                    f"training_image/"
                    + uuid
                    + "/"
                    + video_name.split(".")[0]
                    + "/"
                    + file
                )
                blob_images.upload_from_filename(
                    self.report_path
                    + "low_confidence_detections/"
                    + video_name.split(".")[0]
                    + "/"
                    + file
                )
        else:
            logging.debug(
                "Low confidence detections folder does not exist for video: "
                + video_name
            )

    # To-Do: Implement this function as another step in beam
    def move_folder_gcs(
        self,
        source_bucket: storage.Bucket,
        source_folder_name: str,
        dest_bucket: storage.Bucket,
        dest_folder_name: str,
    ) -> None:
        """
        Moves files from a source GCS "folder" to a destination GCS "folder".

        This method simulates moving a folder by listing blobs in the source path,
        copying each to the destination, and then (implicitly, as deletion is not
        implemented here) removing them from the source. It checks if a blob
        already exists at the destination before copying.

        Note: The original docstring mentioned "source_bucket_name" and "dest_bucket_name",
        but the type hints suggest `storage.Bucket` objects. The implementation uses
        bucket objects. The "Must end with a /." advice for folder names is good practice
        for GCS prefixes.

        Args:
                source_bucket: The source Google Cloud Storage bucket object.
                source_folder_name: The "folder" path (prefix) in the source bucket.
                dest_bucket: The destination Google Cloud Storage bucket object.
                dest_folder_name: The "folder" path (prefix) in the destination bucket.
        """

        blobs = source_bucket.list_blobs(
            prefix=source_folder_name
        )  # List blobs in the source folder

        for blob in blobs:
            # Construct the destination blob name
            relative_path = blob.name[
                len(source_folder_name) :
            ]  # Get the path relative to the source folder
            dest_blob_name = dest_folder_name + relative_path
            # check if the blob is in destination bucket
            if dest_bucket.blob(dest_blob_name).exists():
                print(f"File {blob.name} already exists in {dest_blob_name}.")

            else:
                # Copy the blob to the destination bucket
                new_blob = dest_bucket.copy_blob(blob, dest_bucket, dest_blob_name)

                print(f"File {blob.name} moved to {new_blob.name}.")

    def get_model_name(self) -> str:
        """Returns the name of the YOLO model file."""
        return self.model_name

    def get_iou_threshold(self) -> float:
        """Returns the Intersection over Union (IoU) threshold for NMS."""
        return self.iou_threshold

    def get_confidence_threshold(self) -> float:
        """Returns the confidence threshold for object detection."""
        return self.confidence_threshold

    def get_polygons(self) -> dict:  # Type hint was np.array, but it's a dict of arrays
        """Returns the dictionary of polygons used for counting zones."""
        return self.polygons

    def get_classes(self) -> dict:
        """Returns the dictionary mapping class IDs to class names."""
        return self.classes

    def get_tracker_config(self) -> str:
        """Returns the path to the tracker configuration file."""
        return self.tracker_config

    def set_model_name(self, name: str) -> None:
        """Sets the name of the YOLO model file."""
        self.model_name = name

    def set_iou_threshold(self, iou_threshold: float) -> None:
        """Sets the Intersection over Union (IoU) threshold for NMS."""
        self.iou_threshold = iou_threshold

    def set_confidence_threshold(self, confidence_threshold: float) -> None:
        """Sets the confidence threshold for object detection."""
        self.confidence_threshold = confidence_threshold

    def set_polygons(self, polygons: dict) -> None:  # Type hint was np.array
        """Sets the dictionary of polygons used for counting zones."""
        self.polygons = polygons

    def set_classes(self, classes: dict) -> None:
        """Sets the dictionary mapping class IDs to class names."""
        self.classes = classes

    def set_tracker_config(self, tracker_config: str) -> None:
        """Sets the path to the tracker configuration file."""
        self.tracker_config = tracker_config
