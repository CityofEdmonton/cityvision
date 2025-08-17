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

from train import (
    download_data_from_gcs,
    unzipDataset,
    update_traintxtfile,
    manage_Image_Label_Folder,
    generate_master_yaml,
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

    @patch('train.storage.Client')
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
        mock_bucket.list_blobs.assert_called_once_with(prefix="Dataset/")
        assert mock_blob1.download_to_filename.call_count == 1
        assert mock_blob2.download_to_filename.call_count == 1
        assert mock_blob3.download_to_filename.call_count == 0

    @patch('train.storage.Client')
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
        mock_print.assert_any_call("No zipped files found or downloaded from gs://test-bucket/Dataset/.")

    @patch('train.storage.Client')
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


class TestUnzipDataset:
    """Test cases for unzipDataset function."""

    def test_unzipDataset_success(self):
        """Test successful unzipping of dataset files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test zip file
            test_file_path = os.path.join(temp_dir, "test.zip")
            with zipfile.ZipFile(test_file_path, 'w') as zipf:
                zipf.writestr("test.txt", "test content")
            
            # Create a non-zip file
            non_zip_file = os.path.join(temp_dir, "test.txt")
            with open(non_zip_file, 'w') as f:
                f.write("test")
            
            # Test
            unzipDataset(temp_dir)
            
            # Assertions
            assert not os.path.exists(test_file_path)  # Zip file should be removed
            assert os.path.exists(os.path.join(temp_dir, "test.txt"))  # Content should be extracted
            assert os.path.exists(non_zip_file)  # Non-zip file should remain

    def test_unzipDataset_empty_directory(self):
        """Test unzipping from an empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise any exception
            unzipDataset(temp_dir)


class TestUpdateTrainTxtFile:
    """Test cases for update_traintxtfile function."""

    def test_update_traintxtfile_success(self):
        """Test successful update of train.txt file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test train.txt file
            train_file_path = os.path.join(temp_dir, "train.txt")
            with open(train_file_path, 'w') as f:
                f.write("image1.jpg\nimage2.jpg\nimage3.jpg\n")
            
            # Test
            update_traintxtfile("train.txt", temp_dir)
            
            # Read the updated file
            with open(train_file_path, 'r') as f:
                updated_content = f.read()
            
            # Assertions
            expected_content = f"{os.path.join(temp_dir, 'image1.jpg')}\n{os.path.join(temp_dir, 'image2.jpg')}\n{os.path.join(temp_dir, 'image3.jpg')}\n"
            assert updated_content == expected_content

    def test_update_traintxtfile_empty_filename(self):
        """Test update with empty filename."""
        # Should not raise any exception
        update_traintxtfile("", "test_path")


class TestManageImageLabelFolder:
    """Test cases for manage_Image_Label_Folder function."""

    def test_manage_Image_Label_Folder_with_images_and_labels(self):
        """Test when folder contains images and labels directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create images and labels directories
            images_dir = os.path.join(temp_dir, "images")
            labels_dir = os.path.join(temp_dir, "labels")
            os.makedirs(images_dir)
            os.makedirs(labels_dir)
            
            # Create some test files
            with open(os.path.join(images_dir, "test.jpg"), 'w') as f:
                f.write("test")
            with open(os.path.join(labels_dir, "test.txt"), 'w') as f:
                f.write("test")
            
            # Test
            manage_Image_Label_Folder(temp_dir)
            
            # Assertions
            data_dir = os.path.join(temp_dir, "data")
            assert os.path.exists(os.path.join(data_dir, "images"))
            assert os.path.exists(os.path.join(data_dir, "labels"))
            assert not os.path.exists(images_dir)  # Original should be moved
            assert not os.path.exists(labels_dir)  # Original should be moved

    def test_manage_Image_Label_Folder_without_images_and_labels(self):
        """Test when folder doesn't contain images and labels directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some other directory
            other_dir = os.path.join(temp_dir, "other")
            os.makedirs(other_dir)
            
            # Test
            manage_Image_Label_Folder(temp_dir)
            
            # Assertions
            assert not os.path.exists(os.path.join(temp_dir, "data"))  # No data dir should be created


class TestGenerateMasterYaml:
    """Test cases for generate_master_yaml function."""

    def test_generate_master_yaml_success(self):
        """Test successful generation of master YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test subdirectories with data.yaml files
            subdir1 = os.path.join(temp_dir, "part1")
            subdir2 = os.path.join(temp_dir, "part2")
            os.makedirs(subdir1)
            os.makedirs(subdir2)
            
            # Create data.yaml for part1
            data_yaml1 = {
                "path": ".",
                "Train": "train.txt",
                "Val": "val.txt",
                "names": {0: "person", 1: "car"}
            }
            with open(os.path.join(subdir1, "data.yaml"), 'w') as f:
                yaml.dump(data_yaml1, f)
            
            # Create train.txt for part1
            with open(os.path.join(subdir1, "train.txt"), 'w') as f:
                f.write("image1.jpg\nimage2.jpg\n")
            
            # Create data.yaml for part2
            data_yaml2 = {
                "path": ".",
                "Train": "train.txt",
                "names": {0: "person", 1: "car"}
            }
            with open(os.path.join(subdir2, "data.yaml"), 'w') as f:
                yaml.dump(data_yaml2, f)
            
            # Create train.txt for part2
            with open(os.path.join(subdir2, "train.txt"), 'w') as f:
                f.write("image3.jpg\nimage4.jpg\n")
            
            # Test
            result = generate_master_yaml(temp_dir, "master_data.yaml")
            
            # Assertions
            assert result == os.path.join(temp_dir, "master_data.yaml")
            assert os.path.exists(result)
            
            # Check the content of the generated YAML
            with open(result, 'r') as f:
                master_data = yaml.safe_load(f)
            
            assert master_data["path"] == "../yolo_dataset"
            assert "part1/train.txt" in master_data["train"]
            assert "part2/train.txt" in master_data["train"]
            assert master_data["nc"] == 2
            assert master_data["names"] == {0: "person", 1: "car"}

    def test_generate_master_yaml_no_subfolders(self):
        """Test when no subfolders exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test
            result = generate_master_yaml(temp_dir, "master_data.yaml")
            
            # Assertions
            assert result is None

    def test_generate_master_yaml_no_data_yaml(self):
        """Test when no data.yaml files exist in subfolders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create subdirectory without data.yaml
            subdir = os.path.join(temp_dir, "part1")
            os.makedirs(subdir)
            
            # Test
            result = generate_master_yaml(temp_dir, "master_data.yaml")
            
            # Assertions
            assert result is None


class TestTrainYoloModel:
    """Test cases for train_yolo_model function."""

    @patch('train.YOLO')
    def test_train_yolo_model_success(self, mock_yolo_class):
        """Test successful YOLO model training."""
        # Mock setup
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_model.train.return_value = Mock()
        
        # Test
        train_yolo_model(
            data_yaml_path="test_data.yaml",
            model=mock_model,
            epochs=5,
            img_size=640,
            batch_size=16,
            device="cpu"
        )
        
        # Assertions
        mock_model.train.assert_called_once()
        call_args = mock_model.train.call_args
        assert call_args[1]["data"] == "test_data.yaml"
        assert call_args[1]["epochs"] == 5
        assert call_args[1]["imgsz"] == 640
        assert call_args[1]["batch"] == 16
        assert call_args[1]["device"] == "cpu"
        assert call_args[1]["val"] is False

    @patch('train.YOLO')
    def test_train_yolo_model_with_custom_hsv(self, mock_yolo_class):
        """Test YOLO model training with custom HSV parameters."""
        # Mock setup
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_model.train.return_value = Mock()
        
        # Test
        train_yolo_model(
            data_yaml_path="test_data.yaml",
            model=mock_model,
            epochs=5,
            img_size=640,
            batch_size=16,
            device="cpu",
            hsv_h_range=0.1,
            hsv_s_range=0.6,
            hsv_v_range=0.4
        )
        
        # Assertions
        call_args = mock_model.train.call_args
        assert call_args[1]["hsv_h"] == 0.1
        assert call_args[1]["hsv_s"] == 0.6
        assert call_args[1]["hsv_v"] == 0.4

    @patch('train.YOLO')
    def test_train_yolo_model_exception(self, mock_yolo_class):
        """Test handling of exceptions during training."""
        # Mock setup
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        mock_model.train.side_effect = Exception("Training failed")
        
        # Test
        with patch('builtins.print') as mock_print:
            with pytest.raises(SystemExit):
                train_yolo_model(
                    data_yaml_path="test_data.yaml",
                    model=mock_model,
                    epochs=5,
                    img_size=640,
                    batch_size=16,
                    device="cpu"
                )
        
        # Assertions
        mock_print.assert_any_call("Error during YOLO model training: Training failed")


class TestConstants:
    """Test cases for module constants."""

    def test_constants_are_defined(self):
        """Test that all expected constants are defined."""
        assert GCS_BUCKET_NAME == "open-cityvision"
        assert GCS_DATA_PATH == "Dataset/"
        assert LOCAL_DATA_DIR == "yolo_dataset/"
        assert IMG_SIZE == 640
        assert BATCH_SIZE == 16
        assert DEVICE == "cpu"
        assert EPOCHS == 10


if __name__ == "__main__":
    pytest.main([__file__])
