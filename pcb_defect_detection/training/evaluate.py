import os
import sys
import json
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import training_logger

def evaluate_model(version_dir: str, data_yaml: str):
    """
    Evaluates the model and computes Precision, Recall, mAP50, and mAP50-95.
    Saves outputs as evaluation_metrics.json.
    """
    model_path = os.path.join(version_dir, "best.pt")
    if not os.path.exists(model_path):
        training_logger.error(f"Cannot evaluate: {model_path} does not exist.")
        return

    training_logger.info(f"Starting Evaluation for {model_path}")
    model = YOLO(model_path)
    
    # Ultralytics val automatically generates PR curves and confusion matrices
    # in the runs/detect/val directory (or matching project/name dir)
    metrics = model.val(data=data_yaml, split='val', device='cpu')
    
    results = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "fitness": float(metrics.fitness)
    }
    
    metric_file = os.path.join(version_dir, "evaluation_metrics.json")
    with open(metric_file, "w") as f:
        json.dump(results, f, indent=4)
        
    training_logger.info(f"Evaluation complete. Metrics saved to {metric_file}")
    training_logger.info(f"Results: {results}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py [version_dir] [data_yaml_path]")
        sys.exit(1)
    evaluate_model(sys.argv[1], sys.argv[2])
