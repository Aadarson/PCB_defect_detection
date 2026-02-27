import os
import sys

# Change path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import system_logger

def verify_directories(required_dirs: list) -> bool:
    """Check if mandatory directories exist."""
    success = True
    for directory in required_dirs:
        if not os.path.exists(directory):
            system_logger.error(f"Missing required directory: {directory}")
            success = False
        elif not os.access(directory, os.W_OK):
            system_logger.error(f"Directory exists but is not writable: {directory}")
            success = False
    return success

def verify_dataset(data_yaml_path: str) -> bool:
    """Verify if the YOLO dataset data.yaml file is present."""
    if not os.path.exists(data_yaml_path):
        system_logger.error(f"Dataset data.yaml not found at {data_yaml_path}. Is the dataset downloaded?")
        return False
    return True

def run_preflight_checks():
    """Runs all integrity checks before server start or training."""
    system_logger.info("Starting System Integrity Pre-flight Checks...")
    root_dir = parent_dir
    
    # 1. Directories
    required_dirs = [
        os.path.join(root_dir, "dataset"),
        os.path.join(root_dir, "models"),
        os.path.join(root_dir, "logs"),
        os.path.join(root_dir, "database"),
    ]
    
    # Optional logic: automatically create them if we want, or fail fast.
    # Here, we create them if missing to avoid trivial failures.
    for d in required_dirs:
        os.makedirs(d, exist_ok=True)
    
    if not verify_directories(required_dirs):
        system_logger.error("Directory verification failed. Aborting pre-flight.")
        return False

    # 2. Config Files
    config_yaml = os.path.join(root_dir, "training", "config.yaml")
    if not os.path.exists(config_yaml):
        system_logger.warning(f"training/config.yaml is missing. Training might fail.")
    else:
        system_logger.info("config.yaml is present.")
        
    system_logger.info("Pre-flight checks completed successfully.")
    return True

if __name__ == "__main__":
    if not run_preflight_checks():
        sys.exit(1)
    else:
        sys.exit(0)
