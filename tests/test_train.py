import pytest
import os
import yaml
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock, mock_open
from unittest.mock import call
import zipfile
import sys
from pathlib import Path

# Add the parent directory to the path to import train module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from notebooks.train import (
    download_data_from_gcs,
    unzipDataset,
    train_yolo_model,
    GCS_BUCKET_NAME,
    GCS_DATA_PATH,
    LOCAL_DATA_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
)


class TestDownloadDataFromGCS:
    """Test cases for download_data_from_gcs function."""

    @patch('notebooks.train.storage.Client')
    def test_download_data_from_gcs_success(self, mock_storage_client):
        """Test successful download of data from GCS."""
        # Mock setup
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob1 = Mock()
        mock_blob1.name = "Dataset/file1.zip"
        mock_blob1.endswith.return_value = True
        
        mock_blob2 = Mock()
        mock_blob2.name = "Dataset/file2.zip"
        mock_blob2.endswith.return_value = True
        
        mock_blob3 = Mock()
        mock_blob3.name = "Dataset/directory/"
        mock_blob3.endswith.return_value = False
        
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2, mock_blob3]
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client
        
        # Test
        with patch('os.makedirs'):
            download_data_from_gcs("test-bucket", "Dataset/", "local_dir")
        
        # Assertions
        mock_storage_client.assert_called_once()
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.list_blobs.assert_called_once_with(prefix="Dataset/", delimiter="/")
        assert mock_blob1.download_to_filename.call_count == 1
        assert mock_blob2.download_to_filename.call_count == 1
        assert mock_blob3.download_to_filename.call_count == 0

    @patch('notebooks.train.storage.Client')
    def test_download_data_from_gcs_no_zip_files(self, mock_storage_client):
        """Test when no zip files are found in GCS."""
        # Mock setup
        mock_client = Mock()
        mock_bucket = Mock()
        mock_blob = Mock()
        mock_blob.name = "Dataset/file.txt"
        mock_blob.endswith.return_value = False
        
        mock_bucket.list_blobs.return_value = [mock_blob]
        mock_client.bucket.return_value = mock_bucket
        mock_storage_client.return_value = mock_client
        
        # Test
        with patch('os.makedirs'):
            with patch('builtins.print') as mock_print:
                download_data_from_gcs("test-bucket", "Dataset/", "local_dir")
        
        # Assertions
        expected_message = "No zipped files found or downloaded from gs://test-bucket/Dataset/. Please check bucket name and GCS path, and ensure there are .zip files present."
        mock_print.assert_any_call(expected_message)
        
    @patch('notebooks.train.storage.Client')
    def test_download_data_from_gcs_exception(self, mock_storage_client):
        """Test handling of exceptions during download."""
        # Mock setup
        mock_storage_client.side_effect = Exception("GCS connection failed")
        
        # Test
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit):
                download_data_from_gcs("test-bucket", "Dataset/", "local_dir")
        
        # Assertions
        mock_print.assert_any_call("An error occurred: GCS connection failed")

# class TestUnzipDataset:
#     """Test cases for unzipDataset function."""

#     def test_unzipDataset_success(self):
#         """Test successful unzipping of dataset files."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Create a test zip file
#             test_file_path = os.path.join(temp_dir, "test.zip")
#             with zipfile.ZipFile(test_file_path, 'w') as zipf:
#                 zipf.writestr("test.txt", "test content")
            
#             # Create a non-zip file
#             non_zip_file = os.path.join(temp_dir, "test.txt")
#             with open(non_zip_file, 'w') as f:
#                 f.write("test")
            
#             # Test
#             unzipDataset(temp_dir)
            
#             # Assertions
#             assert not os.path.exists(test_file_path)  # Zip file should be removed
#             assert os.path.exists(os.path.join(temp_dir, "test.txt"))  # Content should be extracted
#             assert os.path.exists(non_zip_file)  # Non-zip file should remain

#     def test_unzipDataset_empty_directory(self):
#         """Test unzipping from an empty directory."""
#         with tempfile.TemporaryDirectory() as temp_dir:
#             # Should not raise any exception
#             unzipDataset(temp_dir)

# class TestTrainYoloModel:
#     """Test cases for train_yolo_model function."""

#     @patch('train.YOLO')
#     def test_train_yolo_model_success(self, mock_yolo_class):
#         """Test successful YOLO model training."""
#         # Mock setup
#         mock_model = Mock()
#         mock_yolo_class.return_value = mock_model
#         mock_model.train.return_value = Mock()
        
#         # Test
#         train_yolo_model(
#             data_yaml_path="test_data.yaml",
#             model=mock_model,
#             epochs=5,
#             img_size=640,
#             batch_size=16,
#             device="cpu"
#         )
        
#         # Assertions
#         mock_model.train.assert_called_once()
#         call_args = mock_model.train.call_args
#         assert call_args[1]["data"] == "test_data.yaml"
#         assert call_args[1]["epochs"] == 5
#         assert call_args[1]["imgsz"] == 640
#         assert call_args[1]["batch"] == 16
#         assert call_args[1]["device"] == "cpu"
#         assert call_args[1]["val"] is False

#     @patch('train.YOLO')
#     def test_train_yolo_model_with_custom_hsv(self, mock_yolo_class):
#         """Test YOLO model training with custom HSV parameters."""
#         # Mock setup
#         mock_model = Mock()
#         mock_yolo_class.return_value = mock_model
#         mock_model.train.return_value = Mock()
        
#         # Test
#         train_yolo_model(
#             data_yaml_path="test_data.yaml",
#             model=mock_model,
#             epochs=5,
#             img_size=640,
#             batch_size=16,
#             device="cpu",
#             hsv_h_range=0.1,
#             hsv_s_range=0.6,
#             hsv_v_range=0.4
#         )
        
#         # Assertions
#         call_args = mock_model.train.call_args
#         assert call_args[1]["hsv_h"] == 0.1
#         assert call_args[1]["hsv_s"] == 0.6
#         assert call_args[1]["hsv_v"] == 0.4

#     @patch('train.YOLO')
#     def test_train_yolo_model_exception(self, mock_yolo_class):
#         """Test handling of exceptions during training."""
#         # Mock setup
#         mock_model = Mock()
#         mock_yolo_class.return_value = mock_model
#         mock_model.train.side_effect = Exception("Training failed")
        
#         # Test
#         with patch('builtins.print') as mock_print:
#             with pytest.raises(SystemExit):
#                 train_yolo_model(
#                     data_yaml_path="test_data.yaml",
#                     model=mock_model,
#                     epochs=5,
#                     img_size=640,
#                     batch_size=16,
#                     device="cpu"
#                 )
        
#         # Assertions
#         mock_print.assert_any_call("Error during YOLO model training: Training failed")

# class TestConstants:
#     """Test cases for module constants."""

#     def test_constants_are_defined(self):
#         """Test that all expected constants are defined."""
#         assert GCS_BUCKET_NAME == "open-cityvision"
#         assert GCS_DATA_PATH == "Dataset/"
#         assert LOCAL_DATA_DIR == "yolo_dataset/"
#         assert IMG_SIZE == 640
#         assert BATCH_SIZE == 16
#         assert DEVICE == "cpu"
#         assert EPOCHS == 10


if __name__ == "__main__":
    pytest.main([__file__])
