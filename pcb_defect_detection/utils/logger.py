import logging
import os
import sys
from io import StringIO
from logging.handlers import RotatingFileHandler

# Define log directories
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Format for logs
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
formatter = logging.Formatter(LOG_FORMAT)

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """
    Function to setup a logger that writes to both console and a specific file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler (Rotating log file, max 5MB, keep 3 backups)
        file_path = os.path.join(LOG_DIR, log_file)
        file_handler = RotatingFileHandler(file_path, maxBytes=5*1024*1024, backupCount=3)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Pre-configured loggers for different components
training_logger = setup_logger("training", "training.log")
api_logger = setup_logger("api", "api.log")
inference_logger = setup_logger("inference", "inference.log")
system_logger = setup_logger("system", "system.log")
