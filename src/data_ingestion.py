import pandas as pd
import os
import sys

sys.path.append('../fraud_detection_system')

from src.logger import get_logger
from src.exception import DataIngestionException

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, file_path):
        """
        Initialize the DataIngestion object with a file path.
    
        """
        self.file_path = file_path
        logger.info(f"DataIngestion initialized with file path: {self.file_path}")

    def load_data(self):
        """
        Load the dataset from a CSV file.
        
        """
        try:
            if not os.path.exists(self.file_path):
                logger.error(f"File not found: {self.file_path}")
                raise DataIngestionException(f"File not found: {self.file_path}")

            logger.info(f"Loading data from {self.file_path}")
            data = pd.read_csv(self.file_path)
            logger.info(f"Data loaded successfully with {data.shape[0]} records and {data.shape[1]} columns.")
            return data

        except Exception as e:
            logger.error(f"An error occurred while loading data: {str(e)}")
            raise DataIngestionException(f"Failed to load data from {self.file_path}", errors=e)

    def validate_data(self, data):
        """
        Validate the loaded dataset by checking for missing values, duplicate entries, and other potential issues.
        
        """
        try:
            logger.info("Starting data validation...")

            # Check for missing values
            missing_values = data.isnull().sum().sum()
            if missing_values > 0:
                logger.warning(f"Data contains {missing_values} missing values. Consider handling them in preprocessing.")

            # Check for duplicate records
            duplicates = data.duplicated().sum()
            if duplicates > 0:
                logger.warning(f"Data contains {duplicates} duplicate records. Consider handling them in preprocessing.")

            # Validate specific columns (e.g., checking that numerical columns contain only valid values)
            if not pd.api.types.is_numeric_dtype(data['amount']):
                logger.error("The 'amount' column contains non-numeric values.")
                raise DataIngestionException("Invalid data type in 'amount' column.")

            logger.info("Data validation completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during data validation: {str(e)}")
            raise DataIngestionException("Failed to validate data.", errors=e)

    def preprocess_data(self, data):
        """
        Perform basic preprocessing on the data, including handling missing values and encoding categorical variables.
        
        """
        try:
            logger.info("Starting data preprocessing...")

            # Handling missing values
            initial_shape = data.shape
            data = data.dropna()  # You can also use data.fillna() to fill missing values
            logger.info(f"Dropped missing values. Data shape changed from {initial_shape} to {data.shape}.")

            # Encoding categorical variables (example: one-hot encoding for 'type', 'branch', 'Acct type', 'Time of day')
            categorical_columns = ['type', 'branch', 'Acct type', 'Time of day']
            data = pd.get_dummies(data, columns=categorical_columns)
            logger.info(f"Categorical columns {categorical_columns} encoded using one-hot encoding.")

            logger.info("Basic preprocessing completed successfully.")
            return data

        except Exception as e:
            logger.error(f"An error occurred during data preprocessing: {str(e)}")
            raise DataIngestionException("Failed to preprocess data.", errors=e)

    def save_preprocessed_data(self, data, output_path):
        """
        Save the preprocessed data to a specified output path.
        
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_csv(output_path, index=False)
            logger.info(f"Preprocessed data saved to {output_path}")

        except Exception as e:
            logger.error(f"An error occurred while saving preprocessed data: {str(e)}")
            raise DataIngestionException("Failed to save preprocessed data.", errors=e)

if __name__ == '__main__':
    # Example usage of the DataIngestion class
    try:
        sys.path.append('../fraud_detection_system')
        ingestion = DataIngestion(file_path='../fraud_detection_system/data/raw/Datasets.csv')
        raw_data = ingestion.load_data()

        # Validate the data
        ingestion.validate_data(raw_data)

        # Preprocess the data
        processed_data = ingestion.preprocess_data(raw_data)

        # Save the preprocessed data
        ingestion.save_preprocessed_data(processed_data, output_path='../fraud_detection_system/data/interim/transaction_data.csv')

    except DataIngestionException as e:
        logger.error(f"Data ingestion process failed: {str(e)}")
