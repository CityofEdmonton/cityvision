import pathlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
import cv2
import time
import supervision as sv
import pandas as pd
from tqdm import tqdm
import logging
import os
import google.cloud.storage as storage



class yolo_counting_model:

	def __init__(self, name: str, config: dict) -> None:
		self.model_name = name
		self.study_name = config["study_name"]
		self.iou_threshold = config["iou_threshold"]
		self.confidence_threshold = config["confidence_threshold"]
		self.polygons = config["polygons"]
		self.classes = config["classes"]
		self.tracker_config = config["tracker_config"]
		self.report_path = config["report_path"]
		self.direction = config["direction"]
		# create dictionary to store crossed objects according to key values
		self.crossed_objects = {self.direction[0]: {}, self.direction[1]: {}}
		self.track_history = defaultdict(lambda: [])
		self.count = 0
		self.model = YOLO(self.model_name)
		#self.model.export(format="onnx", half = False, imgsz=(384,576))
		#self.model = YOLO(self.model_name.split(".")[0]+".onnx")
		try:
			self.model.to('cuda')
		except:
			self.model.export(format="onnx", half = False, imgsz=(384,576))
			self.model = YOLO(self.model_name.split(".")[0]+".onnx")

		if "WB" in self.polygons.keys():
			self.polygons_1 = self.polygons["EB"]
			self.polygons_2 = self.polygons["WB"]
		
		elif "wb" in self.polygons.keys():
			
			self.polygons_1 = self.polygons["eb"]
			self.polygons_2 = self.polygons["wb"]

		else:
			self.polygons_1 = self.polygons["NB"]
			self.polygons_2 = self.polygons["SB"]

		self.classes_ids = [k for k, _ in self.classes.items()]


	def reset(self) -> None:
		self.crossed_objects = {self.direction[0]: {}, self.direction[1]: {}}
		self.track_history = defaultdict(lambda: [])
		self.count = 0

	def run(self, file_path: str, file_name: str) -> None:

		start_time = datetime.strptime(file_name.split("_")[1],"%Y%m%d%H%M")

		cap = cv2.VideoCapture(file_path)
		assert cap.isOpened(), "Error reading video file"
		w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


		logging.info("frame_width: " + str(w))
		logging.info("frame_height: " + str(h))

		frame_generator = sv.get_video_frames_generator(source_path=file_path)

		# Open a video sink for the output video
		video_info = sv.VideoInfo.from_video_path(file_path)
		if not pathlib.Path(self.report_path+"video/").exists():
			pathlib.Path(self.report_path+"video/").mkdir(parents=True, exist_ok=True)
		
		video_report_path = self.report_path+"video/" + file_name.split(".")[0] + ".mp4"
		with sv.VideoSink(video_report_path, video_info) as sink:
			for frame in tqdm(frame_generator, total=video_info.total_frames):
				success, frame = cap.read()
				self.count += 1
				if success and self.count%1 == 0:
					annotated_frame = self.get_count(frame, start_time, file_name.split(".")[0])

					# Draw the line on the frame
					cv2.polylines(annotated_frame, [self.polygons_1], True,(0, 255, 0), 2)
					cv2.polylines(annotated_frame, [self.polygons_2], True,(0, 255, 0), 2)

					# Write the count of objects on each frame
					count_text_1 = f"Objects crossed {self.direction[0]}: {len(self.crossed_objects[self.direction[0]])}"
					count_text_2 = f"Objects crossed {self.direction[1]}: {len(self.crossed_objects[self.direction[1]])}"
					cv2.putText(annotated_frame, count_text_1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
					cv2.putText(annotated_frame, count_text_2, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

					# Write the frame with annotations to the output video
					sink.write_frame(annotated_frame)
				else:
					pass

		# Release the video capture
		cap.release()
		print(f"Data has been written to {self.report_path}")
		print(count_text_1)
		print(count_text_2)

	def get_count(self, frame, start_time, video_name) -> dict:

		results, boxes, class_ids, class_names, annotated_frame = self.get_result(frame, video_name)

		if results[0].boxes.id is not None:
			track_ids = results[0].boxes.id.cpu().int().tolist()

			# Plot the tracks and count objects crossing the line
			for box, track_id, cls in zip(boxes, track_ids, class_names):
					x, y, w, h = box
					pt = (int(x.numpy()), int(y.numpy()))
					cls = cls
					track = self.track_history[track_id]
					track.append((float(x), float(y)))  # x, y center point
					if len(track) > 30:  # retain 30 tracks for 30 frames
							track.pop(0)
					flag_1 = cv2.pointPolygonTest(self.polygons_1,pt,False)
					flag_2 = cv2.pointPolygonTest(self.polygons_2,pt,False)
					# Check if the object crosses the line
					if flag_1 > 0 and flag_1 is not None:  # Assuming objects cross horizontally
							if track_id not in self.crossed_objects[self.direction[0]]:
									time_seen = datetime.fromtimestamp(int(self.count/30) + start_time.timestamp())
									self.crossed_objects[self.direction[0]][track_id] = [time_seen.strftime("%Y-%m-%d %H:%M:%S"), cls]

							# Annotate the object as it crosses the line
							cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

					# Check if the object crosses the line
					if flag_2 > 0 and flag_2 is not None:  # Assuming objects cross horizontally
							if track_id not in self.crossed_objects[self.direction[1]]:
									time_seen = datetime.fromtimestamp(int(self.count/30) + start_time.timestamp())
									self.crossed_objects[self.direction[1]][track_id] = [time_seen.strftime("%Y-%m-%d %H:%M:%S"), cls]

							# Annotate the object as it crosses the line
							cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

					# Annotate center of the object
					cv2.circle(annotated_frame, pt, 5, (0, 255, 0), -1)

		return annotated_frame

	def get_result(self, frame, video_name) -> dict:

		results = self.model.track(frame, classes=self.classes_ids, persist=True, save=False, tracker=self.tracker_config, imgsz=(384,576),
									verbose=False, conf = self.confidence_threshold, iou = self.iou_threshold, agnostic_nms = False)

		# save low confidence detections
		conf_list = results[0].boxes.conf.cpu().float().tolist()
		if any(conf_el < 0.1 for conf_el in conf_list):
			with sv.ImageSink(target_dir_path=self.report_path+"low_confidence_detections/"+video_name,overwrite=False) as sink:
				sink.save_image(image = frame, image_name = f"image_{self.count}.jpg")

		# Get the boxes and track IDs
		boxes = results[0].boxes.xywh.cpu()

		class_ids = results[0].boxes.cls.cpu().int().tolist()

		class_names = [self.classes[i] for i in class_ids]

		# Visualize the results on the frame
		annotated_frame = results[0].plot()

		return (results, boxes, class_ids, class_names, annotated_frame)

	def resample_data(self, df_first: pd.DataFrame, interval: str = '1min') -> pd.DataFrame:

		df = df_first.copy()
		# Reset the index to make the track_id a column
		df.reset_index(inplace=True)
		# Set the index to the timestamp
		df['timestamp'] = pd.to_datetime(df['timestamp'])
		df.set_index('timestamp', inplace=True)
		df.rename(columns={'index': 'Track ID'}, inplace=True)
		# Resample the data by intervals
		resampled_df = df.resample(interval).count()

		# Reset the index to make the time intervals a column
		resampled_df = resampled_df.reset_index()

		# Rename the time interval column to 'timestep'
		resampled_df.rename(columns={'index': 'timestep'}, inplace=True)

		return resampled_df.copy()
	
	def generate_report(self, uuid) -> pd.DataFrame:

		print(self.crossed_objects)

		df_1 = pd.DataFrame.from_dict(self.crossed_objects[self.direction[0]], orient='index', columns=['timestamp', 'Class'])
		df_2 = pd.DataFrame.from_dict(self.crossed_objects[self.direction[1]], orient='index', columns=['timestamp', 'Class'])

		df_1 = self.resample_data(df_1)
		df_2 = self.resample_data(df_2)

		df_1['Direction'] = self.direction[0]
		df_2['Direction'] = self.direction[1]

		resampled_df = pd.concat([df_1, df_2])
		resampled_df['uuid'] = uuid

		print("df", resampled_df)

		return resampled_df
	
	def save_to_gcs(self,storage_client,bucket:str, uuid: str, video_name:str) -> None:
		# save report folder to gcs
		bucket_videos = bucket
		bucket_training_images = storage_client.bucket("processed-cityvision")
		video_name = video_name.split("/")[-1]
		# upload folder to bucket

		blob_video = bucket_training_images.blob(self.study_name + "/annotated_videos/"+video_name)
		blob_video.upload_from_filename(self.report_path+"video/"+video_name)

		self.move_folder_gcs(bucket_videos,self.study_name, bucket_training_images, self.study_name)

		if pathlib.Path(self.report_path+"low_confidence_detections/"+video_name.split(".")[0]).exists():
			for file in os.listdir(self.report_path+"low_confidence_detections/"+video_name.split(".")[0]):
				blob_images = bucket_training_images.blob(f"training_image/"+uuid+"/"+video_name.split(".")[0]+"/"+file)
				blob_images.upload_from_filename(self.report_path+"low_confidence_detections/"+video_name.split(".")[0]+"/"+file)
		else:
			logging.debug("Low confidence detections folder does not exist for video: " + video_name)

	# To-Do: Implement this function as another step in beam
	def move_folder_gcs(self, source_bucket, source_folder_name, dest_bucket, dest_folder_name):
		"""Moves a folder (simulated by copying and deleting) from one GCS bucket to another.

		Args:
			source_bucket_name: The name of the source bucket.
			source_folder_name: The name of the source folder (e.g., "my-folder/").  Must end with a /.
			dest_bucket_name: The name of the destination bucket.
			dest_folder_name: The name of the destination folder (e.g., "moved-folder/"). Must end with a /.

		"""


		blobs = source_bucket.list_blobs(prefix=source_folder_name)  # List blobs in the source folder

		for blob in blobs:
			# Construct the destination blob name
			relative_path = blob.name[len(source_folder_name):] # Get the path relative to the source folder
			dest_blob_name = dest_folder_name + relative_path
			# check if the blob is in destination bucket
			if dest_bucket.blob(dest_blob_name).exists():
				print(f"File {blob.name} already exists in {dest_blob_name}.")
				
			else:
				# Copy the blob to the destination bucket
				new_blob = dest_bucket.copy_blob(blob, dest_bucket, dest_blob_name)

				print(f"File {blob.name} moved to {new_blob.name}.")

	
	def get_model_name(self) -> str:
		return self.model_name

	def get_iou_threshold(self) -> float:
		return self.iou_threshold

	def get_confidence_threshold(self) -> float:
		return self.confidence_threshold

	def get_polygons(self) -> np.array:
		return self.polygons

	def get_classes(self) -> dict:
		return self.classes

	def get_tracker_config(self) -> str:
		return self.tracker_config

	def set_model_name(self, name: str) -> None:
		self.model_name = name

	def set_iou_threshold(self, iou_threshold: float) -> None:
		self.iou_threshold = iou_threshold

	def set_confidence_threshold(self, confidence_threshold: float) -> None:
		self.confidence_threshold = confidence_threshold

	def set_polygons(self, polygons: np.array) -> None:
		self.polygons = polygons

	def set_classes(self, classes: dict) -> None:
		self.classes = classes

	def set_tracker_config(self, tracker_config: str) -> None:
		self.tracker_config = tracker_config

