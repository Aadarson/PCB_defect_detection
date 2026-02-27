import os
import sys
import yaml
import time
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import inference_logger, system_logger

class ModelLoader:
    def __init__(self):
        self.model = None
        self.config = {}
        self.load_config()
        self.load_model()
        
    def load_config(self):
        config_path = os.path.join(parent_dir, "training", "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            system_logger.warning("config.yaml not found. Using inference defaults.")
            self.config = {
                "confidence_threshold": 0.25,
                "iou_threshold": 0.45,
                "device": "cpu"
            }
            
    def load_model(self):
        """Scans models directory for the latest best.pt and loads it efficiently."""
        system_logger.info("Initializing CPU-Optimized Model Loader...")
        models_dir = os.path.join(parent_dir, "models")
        
        model_path = "yolov8n.pt" # Fallback
        
        if os.path.exists(models_dir):
            versions = [d for d in os.listdir(models_dir) if d.startswith("v") and os.path.isdir(os.path.join(models_dir, d))]
            if versions:
                # Get the highest version number
                latest_v = max(versions, key=lambda x: int(x[1:]))
                potential_best = os.path.join(models_dir, latest_v, "best.pt")
                if os.path.exists(potential_best):
                    model_path = potential_best
                    
        system_logger.info(f"Loading weights from: {model_path}")
        try:
            start_t = time.time()
            self.model = YOLO(model_path)
            
            # CRITICAL CPU OPTIMIZATION
            system_logger.info("Fusing model layers for faster CPU inference...")
            self.model.fuse() 
            
            # Optionally force half precision float16 if natively supported (rarely useful on pure CPU, but safe to attempt)
            # self.model.half() # Omitted as it often causes CPU tensor mismatches unless explicitly compiled.
            
            end_t = time.time()
            system_logger.info(f"Model loaded successfully in {round(end_t - start_t, 2)} seconds. Fused: True. Device: CPU")
        except Exception as e:
            system_logger.error(f"Failed to load model: {str(e)}")
            sys.exit(1)
            
    def get_model(self):
        return self.model
        
    def get_thresholds(self):
        return {
            "conf": self.config.get("confidence_threshold", 0.25),
            "iou": self.config.get("iou_threshold", 0.45)
        }

# Singleton instance
loader_instance = ModelLoader()

def get_inference_model():
    return loader_instance.get_model()

def get_inference_thresholds():
    return loader_instance.get_thresholds()
