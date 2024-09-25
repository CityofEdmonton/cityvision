
import apache_beam as beam
import pandas as pd
from google.cloud import storage
import numpy as np
from cityvision.models.main import yolo_counting_model
import pandas as pd
import tempfile
import json



def ProcessVideo(model_input: dict) -> pd.DataFrame:
    """Processes a video and returns the vehicle count."""
    # read from GCS
    storage_client = storage.Client(project="apps-cityvision-prod")
    bucket = storage_client.bucket("dataflow-cityvision")
    blob = bucket.get_blob(model_input["video"])
    file_name = model_input["video"].split("/")[-1]
    uuid = model_input["video"].split("/")[0]
    uuid = uuid.split("_")[0]
    
    # download video
    with tempfile.NamedTemporaryFile() as temp_file:
        blob.download_to_filename(temp_file.name)

        # analyze video
        CONFIDENCE_THRESHOLD = 0.1
        IOU_THRESHOLD = 0.5
        MODEL_NAME = "yolov8l.pt"
        class_names_to_count = {1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        SOURCE = {"EB": np.array(model_input["annotation"]["eb"],np.int32),
                "WB": np.array(model_input["annotation"]["wb"],np.int32)}
        
        Tracker_config_path = "bytetrack.yml"
        Report_path = "report/"

        config = {
            "iou_threshold": IOU_THRESHOLD,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "polygons": SOURCE,
            "classes": class_names_to_count,
            "tracker_config": Tracker_config_path,
            "report_path": Report_path
        }

        model = yolo_counting_model(MODEL_NAME, config)
        model.reset()
        print("Running video: ", model_input["video"])
        model.run(temp_file.name, file_name)
        report = model.generate_report(file_name)

    temp_file.close()

    return report
    

def dataframe_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms a DataFrame to the desired format."""

    df = df.groupby(['uuid','Direction','timestamp']).Class.sum().reset_index()
    df.rename(columns={'Class': 'vehicle_count'}, inplace=True)
    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

# Function to yield dictionaries (necessary for Beam to process DataFrame rows)
def dataframe_to_dicts(dataframe: pd.DataFrame):
    for _, row in dataframe.iterrows():
        yield dict(row)

class BigQueryWriteErrorTransform(beam.PTransform):
    def __init__(self, table_name, schema, retry_strategy=beam.io.gcp.bigquery.RetryStrategy.RETRY_ON_TRANSIENT_ERROR):
        self.table_name = table_name
        self.schema = schema
        self.retry_strategy = retry_strategy
        self.error_schema = error_schema = {
            "fields": [
                {"name": "error_message", "type": "STRING", "mode": "NULLABLE"},
                {"name": "row", "type": "STRING", "mode": "NULLABLE"},
            ]
        }

    def expand(self, pcoll):
        # Write to BigQuery
        _ = pcoll | "WriteToBigQuery" >> beam.io.gcp.bigquery.WriteToBigQuery(
            table=self.table_name,
            schema=self.schema,
            insert_retry_strategy=self.retry_strategy,
            create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
            write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND,
            method=beam.io.gcp.bigquery.WriteToBigQuery.Method.FILE_LOADS  # Avoid Storage API limitations for bad rows
        )