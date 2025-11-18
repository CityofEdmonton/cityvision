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


def unzipDataset(folderPath: str) -> str | None:
    """
    Unzips all .zip files in the specified folder.
    Args:
        folderPath (str): The path to the folder containing .zip files.
    Returns: Name of the folder unzipped
    """
    files = os.listdir(folderPath)
    filename = None
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


def analyze_yolo_dataset(yaml_path: str) -> dict:
    """
    Analyzes a YOLO dataset specified by the data.yaml file to count
    the distribution of objects for each class across all splits,
    including a count of images with no annotations (Background Images).

    This version handles datasets where 'train', 'val', and 'test' point
    to text files containing lists of image paths, relative to the 'path' key.

    Args:
        yaml_path: The file path to the YOLO data.yaml configuration file.

    Returns:
        A dictionary with class IDs as keys and their respective counts as values.
    """
    # 1. Load the YAML configuration file
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"\n❌ Error: data.yaml file not found at '{yaml_path}'")
        return
    except Exception as e:
        print(f"\n❌ Error reading YAML file: {e}")
        return

    # Extract paths and class information
    base_dir = os.path.dirname(yaml_path)
    dataset_root = data.get("path", "")
    train_list_file = data.get("train", "")
    val_list_file = data.get("val", "")
    test_list_file = data.get("test", "")
    class_names = data.get("names", {})  # Expecting dict for class names
    num_classes = data.get("nc", 0)

    if not class_names or not num_classes:
        print("\n❌ Error: 'names' or 'nc' fields missing from data.yaml.")
        return

    print(f"\n--- YOLO Dataset Analyzer ---")
    print(f"Dataset Root: {dataset_root}")
    print(f"Number of Classes (nc): {num_classes}")
    print(f"Class Map: {class_names}")
    print("-" * 35)

    # Dictionary to hold the combined results: {class_id: count}.
    # We will use class_id = -1 to store the count of background images.
    global_distribution = Counter()

    # 2. Define a helper function for processing a directory
    def process_list_file(
        label_dir_key: str, list_file_name: str, dataset_root: str
    ) -> Counter:
        """Reads image list file, derives label paths, and returns class counts."""
        if not list_file_name:
            return Counter()

        list_file_path = os.path.join(base_dir, list_file_name)
        counts = Counter()
        total_images_processed = 0

        if not os.path.isfile(list_file_path):
            print(f"⚠️ Warning: {label_dir_key} list file not found: {list_file_path}")
            return counts

        print(f"\nProcessing {label_dir_key} list file: {list_file_path}")

        try:
            with open(list_file_path, "r") as f:
                image_paths = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Error reading list file {list_file_path}: {e}")
            return counts

        for image_path in image_paths:
            total_images_processed += 1

            # Full path to the image, which includes the 'images' directory
            full_image_path_rel_root = os.path.join(dataset_root, image_path)

            # 1. Get the directory name (e.g., 'path/to/train/images')
            label_dir = os.path.dirname(full_image_path_rel_root)
            # 2. Change the final directory 'images' to 'labels'
            if os.path.basename(label_dir) == "images":
                label_dir = os.path.join(os.path.dirname(label_dir), "labels")

            # 3. Get the filename base (e.g., 'image_001')
            image_filename = os.path.basename(image_path)
            label_file_base = os.path.splitext(image_filename)[0]

            # 4. Construct the full label file path relative to base_dir
            # (base_dir is the directory containing data.yaml)
            label_file_path = os.path.join(
                base_dir,
                label_dir.split(os.path.sep)[-2],
                "labels",
                label_file_base + ".txt",
            )
            objects_in_image_count = 0

            try:
                with open(label_file_path, "r") as f:
                    for line in f:
                        # Class ID is the first token in the line (e.g., '5 0.4410...')
                        class_id_str = line.strip().split(" ")[0]
                        if class_id_str.isdigit():
                            class_id = int(class_id_str)
                            # Only count objects that belong to one of the defined classes (0 to nc-1)
                            if 0 <= class_id < num_classes:
                                counts[class_id] += 1
                                objects_in_image_count += 1
                            else:
                                print(
                                    f"Warning: Found out-of-range class ID {class_id} in {label_file_path}"
                                )

            except FileNotFoundError:
                # File not found means 0 annotated objects
                pass
            except Exception as e:
                # Print the error for other file reading issues
                print(f"Error reading derived label file {label_file_path}: {e}")

            # If the image was processed but yielded no objects (empty or non-existent label file)
            if objects_in_image_count == 0:
                counts[-1] += 1  # Use -1 to count background images

        # Calculate total objects found (excluding background count)
        total_objects_in_split = sum(v for k, v in counts.items() if k != -1)
        print(
            f"Total images processed: {total_images_processed}, Total objects found: {total_objects_in_split} (Excluding backgrounds)"
        )
        return counts

    # 3. Process Train, Val, and Test datasets

    train_counts = process_list_file("Train", train_list_file, dataset_root)
    global_distribution.update(train_counts)

    val_counts = process_list_file("Validation", val_list_file, dataset_root)
    global_distribution.update(val_counts)

    test_counts = process_list_file("Test", test_list_file, dataset_root)
    global_distribution.update(test_counts)

    # 4. Generate and print the final distribution report

    total_background_images = global_distribution.pop(-1, 0)
    total_annotated_objects = sum(global_distribution.values())
    total_items_in_distribution = total_annotated_objects + total_background_images

    if total_items_in_distribution == 0:
        print("\n✅ Analysis Complete: No data found for analysis.")
        return

    print("\n" + "=" * 50)
    print("      FINAL CLASS DISTRIBUTION REPORT")
    print("=" * 50)

    # Format the header
    print(f"{'Class ID':<10} | {'Class Name':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 50)

    # Print Background Images count first
    background_percentage = (
        total_background_images / total_items_in_distribution
    ) * 100
    print(
        f"{'--':<10} | {'Background Images':<20} | {total_background_images:<10} | {background_percentage:.2f}%"
    )
    print("-" * 50)

    # Print annotated objects, sorted by Class ID
    for class_id in sorted(global_distribution.keys()):
        count = global_distribution[class_id]

        # Ensure class_id is within the defined range and handle dict lookup
        # Note: class_names is expected to be a dictionary {id: name}
        # In a typical YOLO file, 'names' is a list/tuple, but the original code expects a dict/list for lookup.
        # Assuming class_names is a list/tuple where index is the class_id:
        try:
            name = class_names[class_id]
        except (TypeError, IndexError):
            # Fallback if class_names is not a list or class_id is out of bounds
            name = f"UNKNOWN CLASS ID {class_id}"

        percentage = (count / total_items_in_distribution) * 100

        print(f"{class_id:<10} | {name:<20} | {count:<10} | {percentage:.2f}%")

    print("-" * 50)
    print(f"{'TOTAL:':<32} | {total_items_in_distribution:<10} | {'100.00%':<10}")
    print("=" * 50)
    return global_distribution, total_items_in_distribution, class_names
