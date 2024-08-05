import os
import urllib.request as request
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig
from pathlib import Path

# Define a class for data ingestion
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion class with a configuration object.

        Args:
            config (DataIngestionConfig): Configuration object for data ingestion.
        """
        self.config = config

    def download_file(self):
        """
        Downloads the file from the specified URL in the configuration if it doesn't already exist locally.
        
        If the file already exists, logs its size.
        """
        # Check if the local data file already exists
        if not os.path.exists(self.config.local_data_file):
            # Download the file from the source URL and save it locally
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            # Log the successful download and headers information
            logger.info(f"{filename} downloaded! with the following info: \n{headers}")
        else:
            # Log that the file already exists and its size
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        """
        Extracts the contents of the zip file specified in the configuration to the unzip directory.

        The unzip directory is created if it doesn't exist.
        """
        # Define the path where the zip file will be extracted
        unzip_path = self.config.unzip_dir
        # Create the unzip directory if it doesn't exist
        os.makedirs(unzip_path, exist_ok=True)
        # Open the zip file and extract its contents to the unzip directory
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
