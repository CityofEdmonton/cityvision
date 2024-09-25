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



class yolo_counting_model:

	def __init__(self, name: str, config: dict) -> None:
		self.model_name = name
		self.iou_threshold = config["iou_threshold"]
		self.confidence_threshold = config["confidence_threshold"]
		self.polygons = config["polygons"]
		self.classes = config["classes"]
		self.tracker_config = config["tracker_config"]
		self.report_path = config["report_path"]
		self.crossed_objects = {"EB": {}, "WB": {}}
		self.track_history = defaultdict(lambda: [])
		self.count = 0
		self.model = YOLO(self.model_name)
		self.model.export(format="onnx", half = False, imgsz=(384,576))
		self.model = YOLO(self.model_name.split(".")[0]+".onnx", task='detect')
		#self.model.to('cuda')
		self.polygons_EB = self.polygons["EB"]
		self.polygons_WB = self.polygons["WB"]
		self.classes_ids = [k for k, _ in self.classes.items()]


	def reset(self) -> None:
		self.crossed_objects = {"EB": {}, "WB": {}}
		self.track_history = defaultdict(lambda: [])
		self.count = 0

	def run(self, file_path: str, file_name: str) -> None:

		start_time = datetime.strptime(file_name.split("_")[1],"%Y%m%d%H%M")

		cap = cv2.VideoCapture(file_path)
		assert cap.isOpened(), "Error reading video file"
		w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


		print("frame_width: ", w)
		print("frame_height: ", h)

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
					annotated_frame = self.get_count(frame, start_time)

					# Draw the line on the frame
					cv2.polylines(annotated_frame, [self.polygons_EB], True,(0, 255, 0), 2)
					cv2.polylines(annotated_frame, [self.polygons_WB], True,(0, 255, 0), 2)

					# Write the count of objects on each frame
					count_text_EB = f"Objects crossed EB: {len(self.crossed_objects['EB'])}"
					count_text_WB = f"Objects crossed WB: {len(self.crossed_objects['WB'])}"
					cv2.putText(annotated_frame, count_text_EB, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

					# Write the frame with annotations to the output video
					sink.write_frame(annotated_frame)
				else:
					pass

		# Release the video capture
		cap.release()
		print(f"Data has been written to {self.report_path}")
		print(count_text_EB)
		print(count_text_WB)

	def get_count(self, frame, start_time) -> dict:

		results, boxes, class_ids,class_names, annotated_frame = self.get_result(frame)

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

					# Check if the object crosses the line
					if cv2.pointPolygonTest(self.polygons_EB,pt,False) > 0:  # Assuming objects cross horizontally
							if track_id not in self.crossed_objects["EB"]:
									time_seen = datetime.fromtimestamp(int(self.count/30) + start_time.timestamp())
									self.crossed_objects["EB"][track_id] = [time_seen.strftime("%Y-%m-%d %H:%M:%S"), cls]

							# Annotate the object as it crosses the line
							cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

					# Check if the object crosses the line
					if cv2.pointPolygonTest(self.polygons_WB,pt,False) > 0:  # Assuming objects cross horizontally
							if track_id not in self.crossed_objects["WB"]:
									time_seen = datetime.fromtimestamp(int(self.count/29) + start_time.timestamp())
									self.crossed_objects["WB"][track_id] = [time_seen.strftime("%Y-%m-%d %H:%M:%S"), cls]

							# Annotate the object as it crosses the line
							cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)

					# Annotate center of the object
					cv2.circle(annotated_frame, pt, 5, (0, 255, 0), -1)

		return annotated_frame

	def get_result(self, frame) -> dict:

		results = self.model.track(frame, classes=self.classes_ids, persist=True, save=False, tracker=self.tracker_config, imgsz=(384,576),
									verbose=False, conf = self.confidence_threshold, iou = self.iou_threshold, agnostic_nms = False)

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
	
	def generate_report(self, name) -> pd.DataFrame:

		df_EB = pd.DataFrame.from_dict(self.crossed_objects["EB"], orient='index', columns=['timestamp', 'Class'])
		df_WB = pd.DataFrame.from_dict(self.crossed_objects["WB"], orient='index', columns=['timestamp', 'Class'])

		df_EB = self.resample_data(df_EB)
		df_WB = self.resample_data(df_WB)

		df_EB['Direction'] = 'EB'
		df_WB['Direction'] = 'WB'

		resampled_df = pd.concat([df_EB, df_WB])
		resampled_df['uuid'] = name  # for testing let it be the name of the video

		return resampled_df
		#if not pathlib.Path(self.report_path + "result/").exists():
		#	pathlib.Path(self.report_path+"result/").mkdir(parents=True, exist_ok=True)
		#
		#file_name = self.report_path+"result/" + name.split(".")[0] + ".csv"
		#
		#resampled_df.to_csv(file_name, index=False)

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

