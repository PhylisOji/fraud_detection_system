# src/pipeline.py
import sys

sys.path.append('../fraud_detection_system')

from src.data_ingestion import DataIngestion
from src.feature_engineering import FeatureEngineering
from src.data_modelling import DataModelling
import os

def run_pipeline(data_file, model_output_path):
    # Step 1: Data Ingestion
    ingestion = DataIngestion(file_path=data_file)
    data = ingestion.load_data()

    # Debugging: Print out the columns in the dataset
    print("Columns in the dataset:", data.columns)

    # Step 2: Feature Engineering
    engineering = FeatureEngineering()
    data = engineering.check_and_handle_missing_values(data)
    data = engineering.create_features(data)
    # Drop unnecessary columns
    columns_to_drop = ['Unnamed: 0', 'nameOrig', 'nameDest', 'Date of transaction', 'Time of day', 'transaction_date']
    data = data.drop(columns=columns_to_drop)

    # Adjust categorical columns based on the actual columns in the dataset
    categorical_columns = ['type', 'branch', 'Acct type']  # Make sure these match exactly with data.columns
    data = engineering.encode_categorical(data, categorical_columns)  # Encode categorical columns

    # Step 3: Model Training
    modelling = DataModelling()
    X = data.drop(columns=['isFraud'])  # Features
    y = data['isFraud']                # Target

    results = modelling.train_and_evaluate_models(X, y)

    # Step 4: Save the best model
    best_model_name = max(results, key=lambda name: results[name]['accuracy'])
    best_model_path = f'../fraud_detection_system/models/pipeline_{best_model_name.replace(" ", "_").lower()}.pkl'
    modelling.save_best_model(model_path=best_model_path)

    print(f"Best model ({best_model_name}) saved to {model_output_path}.")

if __name__ == '__main__':
    data_file = '../fraud_detection_system/data/raw/Datasets.csv'
    model_output_path = '../fraud_detetection_system/models/fraud_detection_model.pkl'
    run_pipeline(data_file, model_output_path)
