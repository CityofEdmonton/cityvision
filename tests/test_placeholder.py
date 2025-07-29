import pytest
import os
import shutil
import sys
from train import download_data_from_gcs
# --- Pytest Fixtures leveraging 'mocker' ---

@pytest.fixture
def mock_gcs_client(mocker):
    """
    Fixture to mock google.cloud.storage.Client using pytest-mock's mocker.
    The 'mocker' fixture is provided by pytest-mock.
    """
    # mocker.patch() is equivalent to unittest.mock.patch() but with pytest's
    # automatic teardown for mocks.
    mock_client = mocker.patch('google.cloud.storage.Client')
    mock_bucket = mocker.Mock() # mocker.Mock() is also available, or plain unittest.mock.Mock()
    mock_client.return_value.bucket.return_value = mock_bucket
    return mock_client, mock_bucket

@pytest.fixture
def mock_sys_exit(mocker):
    """
    Fixture to mock sys.exit using pytest-mock's mocker.
    """
    return mocker.patch('sys.exit')

@pytest.fixture
def capsys_output(capsys):
    """
    Fixture to capture stdout/stderr. 'capsys' is a built-in pytest fixture,
    not specifically from pytest-mock, but commonly used alongside mocking.
    """
    return capsys

# --- Pytest Test Functions ---

def test_successful_download_of_zip_files(mock_gcs_client, tmp_path, capsys_output):
    """Test that only .zip files are downloaded successfully."""
    mock_client, mock_bucket = mock_gcs_client
    mock_bucket_name = "test-bucket"
    mock_gcs_path = "data/zipped_files/"
    local_dir = tmp_path / "test_downloads" # tmp_path provides a unique temp dir

    # Create mock blob objects. mocker.Mock() can be used here too.
    mock_blob_zip1 = mock_client.Mock() # Using mock_client.Mock() is also an option if you like
    mock_blob_zip1.name = "data/zipped_files/file1.zip"
    mock_blob_zip1.download_to_filename = mock_client.Mock()

    mock_blob_zip2 = mock_client.Mock()
    mock_blob_zip2.name = "data/zipped_files/subfolder/file2.zip"
    mock_blob_zip2.download_to_filename = mock_client.Mock()

    mock_blob_txt = mock_client.Mock()
    mock_blob_txt.name = "data/zipped_files/readme.txt"
    mock_blob_txt.download_to_filename = mock_client.Mock()

    mock_blob_dir = mock_client.Mock()
    mock_blob_dir.name = "data/zipped_files/subfolder/"
    mock_blob_dir.download_to_filename = mock_client.Mock()

    mock_bucket.list_blobs.return_value = [
        mock_blob_zip1,
        mock_blob_zip2,
        mock_blob_txt,
        mock_blob_dir
    ]

    download_data_from_gcs(mock_bucket_name, mock_gcs_path, str(local_dir)) # Pass string path

    # Assertions using standard 'assert' and mock methods
    mock_client.assert_called_once()
    mock_client.return_value.bucket.assert_called_once_with(mock_bucket_name)
    mock_bucket.list_blobs.assert_called_once_with(prefix=mock_gcs_path)

    mock_blob_zip1.download_to_filename.assert_called_once_with(
        os.path.join(str(local_dir), "file1.zip")
    )
    mock_blob_zip2.download_to_filename.assert_called_once_with(
        os.path.join(str(local_dir), "file2.zip")
    )
    assert not mock_blob_txt.download_to_filename.called # Using .called for not called
    assert not mock_blob_dir.download_to_filename.called

    assert local_dir.is_dir() # pytest's tmp_path returns a Path object

    captured = capsys_output.readouterr()
    # You can assert on specific print messages if the function prints success messages
    # assert "Successfully downloaded 2 zipped files from GCS." in captured.out


def test_no_zip_files_found(mock_gcs_client, tmp_path, capsys_output):
    """Test the scenario where no .zip files are found."""
    mock_client, mock_bucket = mock_gcs_client
    mock_bucket_name = "test-bucket"
    mock_gcs_path = "data/empty_dir/"
    local_dir = tmp_path / "test_downloads"

    mock_bucket.list_blobs.return_value = []

    download_data_from_gcs(mock_bucket_name, mock_gcs_path, str(local_dir))

    captured = capsys_output.readouterr()
    assert f"No zipped files found or downloaded from gs://{mock_bucket_name}/{mock_gcs_path}." in captured.out
    assert local_dir.is_dir()


def test_gcs_exception_handling(mock_gcs_client, mock_sys_exit, tmp_path, capsys_output):
    """Test that exceptions from GCS operations are caught and handled."""
    mock_client, _ = mock_gcs_client
    mock_bucket_name = "test-bucket"
    mock_gcs_path = "data/error_path/"
    local_dir = tmp_path / "test_downloads"

    # Set the side_effect on the mocked Client to raise an exception
    mock_client.side_effect = Exception("Mock GCS connection error")

    download_data_from_gcs(mock_bucket_name, mock_gcs_path, str(local_dir))

    captured = capsys_output.readouterr()
    assert "An error occurred: Mock GCS connection error" in captured.out
    assert "Error downloading data from GCS: Mock GCS connection error" in captured.out
    assert "Please ensure your Google Cloud credentials are set up correctly." in captured.out

    # Assert that sys.exit(1) was called via the mocked sys.exit
    mock_sys_exit.assert_called_once_with(1)


def test_gcs_list_blobs_exception(mock_gcs_client, mock_sys_exit, tmp_path, capsys_output):
    """Test exception during list_blobs call."""
    _, mock_bucket = mock_gcs_client # We only need mock_bucket here
    mock_bucket_name = "test-bucket"
    mock_gcs_path = "data/error_listing/"
    local_dir = tmp_path / "test_downloads"

    # Set the side_effect on the mocked list_blobs method
    mock_bucket.list_blobs.side_effect = Exception("Mock list_blobs error")

    download_data_from_gcs(mock_bucket_name, mock_gcs_path, str(local_dir))

    captured = capsys_output.readouterr()
    assert "An error occurred: Mock list_blobs error" in captured.out
    assert "Error downloading data from GCS: Mock list_blobs error" in captured.out

    mock_sys_exit.assert_called_once_with(1)


def test_gcs_download_to_filename_exception(mock_gcs_client, mock_sys_exit, tmp_path, capsys_output):
    """Test exception during download_to_filename call."""
    _, mock_bucket = mock_gcs_client # We only need mock_bucket here
    mock_bucket_name = "test-bucket"
    mock_gcs_path = "data/error_downloading/"
    local_dir = tmp_path / "test_downloads"

    # Create a mock blob and set its download_to_filename side_effect
    mock_blob_zip = mock_gcs_client[0].Mock() # Access the MockerFixture's Mock() method
    mock_blob_zip.name = "data/error_downloading/corrupt.zip"
    mock_blob_zip.download_to_filename.side_effect = Exception("Mock download error")

    mock_bucket.list_blobs.return_value = [mock_blob_zip]

    download_data_from_gcs(mock_bucket_name, mock_gcs_path, str(local_dir))

    captured = capsys_output.readouterr()
    assert "An error occurred: Mock download error" in captured.out
    assert "Error downloading data from GCS: Mock download error" in captured.out

    mock_sys_exit.assert_called_once_with(1)