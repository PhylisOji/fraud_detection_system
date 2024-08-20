import sys

sys.path.append('../fraud_detection_system')

from src.logger import get_logger

logger = get_logger(__name__)

class FraudDetectionException(Exception):
    """ Basic Exception for all the other errors raised in this project"""
    def __init__(self,message=None,errors=None):
        super().__init__(message)
        self.errors=errors
        if message:
            #Log the Error Message
            logger.error(message)
        if errors:
            #Log the Error Details
            logger.error(errors)

class DataIngestionException(FraudDetectionException):
    """Exception raised in the Data Ingestion Process"""
    def __init__(self, message="Error occured during the Data Ingestion Process", errors=None):
        super().__init__(message, errors)

class DataModellingException(FraudDetectionException):
    """Exception raised in the Data Modelling Process"""
    def __init__(self, message="Error occured during the Data Modelling Process", errors=None):
        super().__init__(message, errors)

class FeatureEngineeringException(FraudDetectionException):
    """Exception raised in the Feature Engineering Process"""
    def __init__(self, message="Error occured during the Feature Engineering Process", errors=None):
        super().__init__(message, errors)

class PipelineException(FraudDetectionException):
    """Exception raised in the Pipeline Process"""
    def __init__(self, message="Error occured during the Pipeline Process", errors=None):
        super().__init__(message, errors)