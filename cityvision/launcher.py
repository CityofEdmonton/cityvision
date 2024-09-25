import argparse
import ndjson
from google.cloud import storage
from cityvision import my_pipeline
from typing import List, Union, Optional

# for python 3.9 if change to python >=3.10 need to change typing hint
def run(argv: Optional[List[str]] = None):
    """Parses the parameters provided on the command line and runs the pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", required=True, help="Input video folder")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--annotation_folder", required=True, help="Annotation folder")
    parser.add_argument("--output_table", required=True, help="Output table")

    pipeline_args, other_args = parser.parse_known_args(argv)

    # read video files from GCS
    storage_client = storage.Client(project="apps-cityvision-prod")
    print(pipeline_args.video_folder)
    bucket = storage_client.bucket("dataflow-cityvision")
    bl = bucket.list_blobs(prefix=pipeline_args.video_folder)
    videos_folder_list = [blob.name for blob in bl if blob.name.endswith(".mp4")]
        
    # read json files from GCS
    blob = bucket.get_blob(pipeline_args.annotation_folder)

    # convert to string
    json_data_string = blob.download_as_string()
    json_data = ndjson.loads(json_data_string)
    annotation_array = json_data[0]  # expected fomat: [{"eb": array of coordinates, "wb": array of coordinates}]
    model_input = [ {"video": video, "annotation": annotation_array} for video in videos_folder_list]
    pipeline = my_pipeline.vechile_counting(
        model_input, pipeline_args.output_table, other_args
    )