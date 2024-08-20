import pandas as pd
import sys

sys.path.append('../fraud_detection_system')

from src.data_modelling import DataModelling
from src.logger import get_logger
from src.exception import DataModellingException

logger = get_logger(__name__)

def main():
    try:
        logger.info("Starting the model training process...")

        # Load the preprocessed and engineered data
        sys.path.append('../fraud_detection_system')
        data_path = '../fraud_detection_system/data/processed/processed_transaction_data.csv'
        data = pd.read_csv(data_path)

        # Drop unnecessary columns
        columns_to_drop = ['Unnamed: 0', 'nameOrig', 'nameDest', 'Date of transaction', 'Time of day', 'transaction_date']
        data = data.drop(columns=columns_to_drop)

        # Separate features and target
        X = data.drop(columns=['isFraud'])
        y = data['isFraud']

        # Initialize the DataModelling class with default models
        modelling = DataModelling()

        # Train and evaluate models
        results = modelling.train_and_evaluate_models(X, y)

        # Save the best model based on accuracy
        best_model_name = max(results, key=lambda name: results[name]['accuracy'])
        best_model_path = f'../fraud_detection_system/models/{best_model_name.replace(" ", "_").lower()}.pkl'
        modelling.save_best_model(model_path=best_model_path)

        logger.info(f"Model training process completed successfully.")
        logger.info(f"Best model: {best_model_name} with accuracy: {results[best_model_name]['accuracy']:.4f}")

    except DataModellingException as e:
        logger.error(f"Model training process failed: {str(e)}")

if __name__ == '__main__':
    main()
