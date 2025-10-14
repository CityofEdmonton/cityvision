import os
from google.cloud import storage

import zipfile


def download_data_from_gcs(bucket_name: str, gcs_path: str, local_dir: str) -> None:
    """
    Downloads a directory and its contents from a Google Cloud Storage bucket.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_path (str): The path to the directory in GCS (e.g., "datasets/my_data/").
        local_dir (str): The local directory to save the downloaded files.
    """
    # if the gcs_path is just ".", we want to list all blobs in the bucket
    if gcs_path == "" or gcs_path == ".":
        prefix = ""
    else:
        # Ensure the prefix ends with a '/' to treat it as a directory
        prefix = gcs_path.rstrip("/") + "/"

    print(
        f"Attempting to download data from gs://{bucket_name}/{gcs_path} to {local_dir}"
    )

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Ensure local directory exists
        os.makedirs(local_dir, exist_ok=True)

        blobs = bucket.list_blobs(
            prefix=prefix, delimiter="/"
        )  # List all blobs with the given prefix
        downloaded_count = 0
        for blob in blobs:
            # Check if the blob is a .zip file and not a directory itself
            if blob.name.endswith(".zip"):
                # Construct local file path, taking only the base file name
                local_file_name = os.path.basename(blob.name)
                local_file_path = os.path.join(local_dir, local_file_name)

                print(f"Downloading {blob.name} to {local_file_path}")
                blob.download_to_filename(local_file_path)
                downloaded_count += 1
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
        if not file.endswith(".zip"):
            continue
        filename = file.split(".")[0]
        fullPath = os.path.join(folderPath, file)
        if os.path.isfile(fullPath):
            with zipfile.ZipFile(fullPath, "r") as zip_ref:
                zip_ref.extractall(os.path.join(folderPath, filename))

        os.remove(fullPath)
    return filename
