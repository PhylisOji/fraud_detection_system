from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import sys
import os

sys.path.append('../fraud_detection_system')

from src.logger import get_logger
from src.exception import DataModellingException

logger = get_logger(__name__)

class DataModelling:
    def __init__(self, models=None):
        """
        Initialize the DataModelling class with a dictionary of models.
        If no models are provided, use the default set of classification models.
        """
        self.models = models if models else self.get_default_models()
        self.results = {}

    def get_default_models(self):
        """
        Return a dictionary of default classification models.
        """
        return {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "k-Neighbors Classifier": KNeighborsClassifier(),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Random Forest Classifier": RandomForestClassifier(random_state=42)
        }

    def train_and_evaluate_models(self, X, y, test_size=0.2):
        """
        Train and evaluate all models in the self.models dictionary.
        Store the results for each model including accuracy, classification report, and confusion matrix.
        """
        try:
            logger.info("Starting the training and evaluation of models...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            for model_name, model in self.models.items():
                logger.info(f"Training {model_name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                report = classification_report(y_test, y_pred)
                confusion = confusion_matrix(y_test, y_pred)

                logger.info(f"{model_name} completed with accuracy: {accuracy:.4f}")
                logger.info(f"Classification Report for {model_name}:\n{report}")
                logger.info(f"Confusion Matrix for {model_name}:\n{confusion}\n")

                self.results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'classification_report': report,
                    'confusion_matrix': confusion
                }

            return self.results

        except Exception as e:
            logger.error(f"An error occurred during model training and evaluation: {str(e)}")
            raise DataModellingException("Failed to train and evaluate models.", errors=e)

    def save_best_model(self, model_path='../fraud_detection_system/models/fraud_detection_model.pkl'):
        """
        Save the best model based on accuracy to the specified file path.
        """
        try:
            # Identify the best model based on accuracy
            best_model_name = max(self.results, key=lambda name: self.results[name]['accuracy'])
            best_model = self.results[best_model_name]['model']

            logger.info(f"Best model identified: {best_model_name} with accuracy: {self.results[best_model_name]['accuracy']:.4f}")

            # Ensure the directory exists
            directory = os.path.dirname(model_path)
            if not os.path.exists(directory):
                logger.info(f"Creating directory {directory}...")
                os.makedirs(directory, exist_ok=True)

            # Save the best trained model
            logger.info(f"Saving the best model to {model_path}...")
            joblib.dump(best_model, model_path)
            logger.info("Best model saved successfully.")

        except Exception as e:
            logger.error(f"An error occurred while saving the best model: {str(e)}")
            raise DataModellingException("Failed to save the best model.", errors=e)

    def load_model(self, model_path):
        """
        Load a pre-trained model from the specified file path.
        """
        try:
            logger.info(f"Loading model from {model_path}...")
            model = joblib.load(model_path)
            logger.info("Model loaded successfully.")
            return model

        except Exception as e:
            logger.error(f"An error occurred while loading the model: {str(e)}")
            raise DataModellingException("Failed to load the model.", errors=e)
