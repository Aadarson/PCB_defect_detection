import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import system_logger

def validate_deployment_ready(version_dir: str) -> bool:
    """Verifies that a model version has all research-grade and deployment-ready artifacts."""
    system_logger.info(f"Initiating final validation for model version in folder: {version_dir}")
    
    if not os.path.exists(version_dir):
        system_logger.error(f"Version directory {version_dir} does not exist.")
        return False
        
    required_files = [
        "best.pt",
        "dataset_hash.txt",
        "benchmark.json",
        "final_report.json",
        "config_snapshot.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        f_path = os.path.join(version_dir, file)
        if not os.path.exists(f_path):
            missing_files.append(file)
            system_logger.error(f"Validation Error: Mandatory file {file} is missing.")
            
    if missing_files:
        system_logger.error(f"Deployment Validation Failed. Missing {len(missing_files)} components.")
        return False
        
    system_logger.info(f"Model version at {version_dir} has passed all integrity checks. Deployment ready.")
    return True

if __name__ == "__main__":
    # Test block
    if len(sys.argv) < 2:
        print("Usage: python final_validation.py [path_to_model_version_folder]")
        sys.exit(1)
        
    version_dir = sys.argv[1]
    if not validate_deployment_ready(version_dir):
        sys.exit(1)
    sys.exit(0)
