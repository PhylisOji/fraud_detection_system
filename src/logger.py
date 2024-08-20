import logging
import os

#Directory to store the logging file
LOG_DIR = "artifacts"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

#Configure the Logging Path
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,"fraud_detection.logs")),
        logging.StreamHandler()
    ]
)

def get_logger(name):
    """Get a logger instance with a given name"""
    return logging.getLogger(name)