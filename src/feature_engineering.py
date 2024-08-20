import pandas as pd
import numpy as np
import sys
import os
from sklearn.impute import SimpleImputer

sys.path.append('../fraud_detection_system')

from src.logger import get_logger
from src.exception import FeatureEngineeringException

logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self):
        logger.info("FeatureEngineering initialized.")
    
    def check_and_handle_missing_values(self, data):
        """
        Check for missing values and handle them separately for numeric and categorical columns.
        """
        try:
            logger.info("Checking for and handling missing values...")
            
            # Separate numeric and categorical columns
            numeric_data = data.select_dtypes(include=[np.number])
            categorical_data = data.select_dtypes(exclude=[np.number])

            # Handle missing values in numeric columns with median
            if numeric_data.isnull().any().any():
                numeric_imputer = SimpleImputer(strategy='median')
                numeric_data = pd.DataFrame(numeric_imputer.fit_transform(numeric_data), columns=numeric_data.columns)
                logger.info("Handled missing values in numeric columns with median strategy.")

            # Handle missing values in categorical columns with most frequent value
            if categorical_data.isnull().any().any():
                categorical_imputer = SimpleImputer(strategy='most_frequent')
                categorical_data = pd.DataFrame(categorical_imputer.fit_transform(categorical_data), columns=categorical_data.columns)
                logger.info("Handled missing values in categorical columns with most frequent strategy.")

            # Combine the numeric and categorical data back together
            data = pd.concat([numeric_data, categorical_data], axis=1)

            return data

        except Exception as e:
            logger.error(f"Error in handling missing values: {str(e)}")
            raise FeatureEngineeringException("Failed to handle missing values.", errors=e)

    def create_features(self, data):
        """
        Create time-based and amount-based features.
        """
        try:
            logger.info("Creating features...")

            # Time-based features
            data['transaction_date'] = pd.to_datetime(data['Date of transaction'], format='%d/%m/%Y')
            data['day_of_week'] = data['transaction_date'].dt.dayofweek

            # Amount-based features
            data['log_amount'] = np.log1p(data['amount'])
            data['orig_balance_diff'] = data['oldbalanceOrg'] - data['newbalanceOrig']
            data['dest_balance_diff'] = data['oldbalanceDest'] - data['newbalanceDest']

            logger.info("Features created successfully.")
            return data
        except Exception as e:
            logger.error(f"Error in creating features: {str(e)}")
            raise FeatureEngineeringException("Failed to create features.", errors=e)

    def encode_categorical(self, data, categorical_columns):
        """
        Encode categorical features using one-hot encoding.
        """
        try:
            logger.info("Encoding categorical features...")
            data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
            logger.info("Categorical features encoded successfully.")
            return data
        except Exception as e:
            logger.error(f"Error in encoding categorical features: {str(e)}")
            raise FeatureEngineeringException("Failed to encode categorical features.", errors=e)

    def process_data(self, data, categorical_columns):
        """
        Full data processing pipeline: missing value handling, feature creation, and encoding.
        """
        data = self.check_and_handle_missing_values(data)
        data = self.create_features(data)
        data = self.encode_categorical(data, categorical_columns)
        return data

    def save_data(self, data, output_path):
        """
        Save the processed dataset to a specified path.
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            data.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error in saving data: {str(e)}")
            raise FeatureEngineeringException("Failed to save processed data.", errors=e)

if __name__ == '__main__':
    try:
        data_path = '../fraud_detection_system/data/raw/Datasets.csv'
        engineered_data_path = '../fraud_detection_system/data/processed/processed_transaction_data.csv'

        # Load and process data
        data = pd.read_csv(data_path)
        feature_engineering = FeatureEngineering()

        categorical_columns = ['type', 'branch', 'Acct type']  # Define categorical columns

        processed_data = feature_engineering.process_data(data, categorical_columns)

        # Save the processed data
        feature_engineering.save_data(processed_data, output_path=engineered_data_path)

    except FeatureEngineeringException as e:
        logger.error(f"Feature engineering process failed: {str(e)}")