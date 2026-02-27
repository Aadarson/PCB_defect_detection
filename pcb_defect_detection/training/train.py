import os
import sys
import yaml
import torch
import random
import numpy as np
import psutil
from datetime import datetime
from ultralytics import YOLO

# Path setup to access utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import training_logger
from utils.class_analysis import analyze_class_distribution, generate_dataset_hash

def check_memory(limit_percent):
    mem = psutil.virtual_memory()
    return mem.percent > limit_percent

def set_deterministic_seed(seed=42):
    training_logger.info(f"Setting deterministic seed to {seed}...")
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Advanced torch reproducibility flags for CPU
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

def train_yolo():
    config_path = os.path.join(current_dir, "config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 1. Reproducibility Control
    if config.get("deterministic_mode", False):
        set_deterministic_seed(config.get("seed", 42))

    # 2. Versioning Setup
    models_dir = os.path.join(parent_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    existing_versions = [d for d in os.listdir(models_dir) if d.startswith("v") and os.path.isdir(os.path.join(models_dir, d))]
    
    next_version_num = 1
    if existing_versions:
        nums = [int(v[1:]) for v in existing_versions if v[1:].isdigit()]
        if nums:
            next_version_num = max(nums) + 1
            
    version_dir = os.path.join(models_dir, f"v{next_version_num}")
    os.makedirs(version_dir, exist_ok=True)
    
    training_logger.info(f"Initiating Training Run -> Version: v{next_version_num}")

    # 3. Analyze Data Hash & Imbalance
    dataset_dir = os.path.join(parent_dir, "dataset")
    data_yaml = os.path.join(dataset_dir, "data.yaml")
    labels_dir_a = os.path.join(dataset_dir, "labels", "train")
    labels_dir_b = os.path.join(dataset_dir, "train", "labels")
    labels_dir = labels_dir_b if os.path.exists(labels_dir_b) else labels_dir_a
    
    if os.path.exists(labels_dir):
        # Generate Hash
        dataset_hash = generate_dataset_hash(dataset_dir)
        with open(os.path.join(version_dir, "dataset_hash.txt"), "w") as f:
             f.write(dataset_hash)
             
        # Analyze Imbalance
        # Note: In ultralytics training loop, dynamic class weights require custom callback interception.
        # We process the analysis here for the final report. 
        # Ultralytics auto-balances bounding boxes usually, but we flag if > 40% dominance
        analysis = analyze_class_distribution(labels_dir, num_classes=6, threshold=config.get("imbalance_threshold", 0.40))
        if analysis["imbalance_detected"]:
            training_logger.warning("Class imbalance > threshold. Strongly consider gathering more minority class data.")

    # Save Config Snapshot
    with open(os.path.join(version_dir, "config_snapshot.yaml"), "w") as f:
        yaml.dump(config, f)

    # 4. Model & Resumption Logic
    model_path = 'yolov8n.pt'
    if config.get('resume_training') and next_version_num > 1:
        prev_best = os.path.join(models_dir, f"v{next_version_num-1}", "best.pt")
        if os.path.exists(prev_best):
            training_logger.info(f"Resuming weights from {prev_best}")
            model_path = prev_best
            
    model = YOLO(model_path)
    
    # Ultralytics args mapping
    train_args = {
        "data": data_yaml,
        "epochs": config['epochs'],
        "batch": config['batch_size'],
        "imgsz": config['image_size'],
        "device": config['device'],
        "workers": config['workers'],
        "cache": config['cache'],
        "patience": config['early_stopping_patience'],
        "optimizer": config['optimizer'],
        "lr0": config['learning_rate'],
        "weight_decay": config['weight_decay'],
        "project": models_dir,
        "name": f"v{next_version_num}",
        "exist_ok": True, # Force into our version dir
        # Augmentation Overrides
        "hsv_h": config['augmentation']['hsv_h'],
        "hsv_s": config['augmentation']['hsv_s'],
        "hsv_v": config['augmentation']['hsv_v'],
        "degrees": config['augmentation']['degrees'],
        "translate": config['augmentation']['translate'],
        "scale": config['augmentation']['scale'],
        "mosaic": config['augmentation']['mosaic'],
        "deterministic": config['deterministic_mode'],
        "val": True
    }

    # Custom Memory Guards via Callbacks
    def on_train_epoch_end(trainer):
        # Callback for ultralytics to check memory
        if config.get('enable_memory_guard'):
            if check_memory(config.get('memory_limit_percent', 85)):
                training_logger.error("Memory threshold exceeded! Initiating safe abort.")
                trainer.stop = True
                
        # Simple Overfitting Monitor concept
        current_epoch_val_loss = trainer.metrics.get('val/box_loss', 0)
        current_epoch_train_loss = trainer.loss.item() if hasattr(trainer, 'loss') else 0
        if current_epoch_train_loss < 1.0 and current_epoch_val_loss > getattr(trainer, 'last_val_loss', 99.0):
            getattr(trainer, 'overfitting_cnt', 0)
            trainer.overfitting_cnt += 1
            if trainer.overfitting_cnt > 3:
                training_logger.warning(f"Overfitting strongly detected for 3 epochs. Val box loss rising: {current_epoch_val_loss}")
        else:
            trainer.overfitting_cnt = 0
            
        trainer.last_val_loss = current_epoch_val_loss
        training_logger.info(f"Epoch {trainer.epoch + 1}/{trainer.epochs} Complete. Memory: {psutil.virtual_memory().percent}%")

    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    training_logger.info("Beginning YOLO Model Training...")
    try:
        results = model.train(**train_args)
    except Exception as e:
        training_logger.error(f"Training crashed unexpectedly: {str(e)}")
        sys.exit(1)

    # Ensure best.pt is isolated to version folder
    best_weights = os.path.join(models_dir, f"v{next_version_num}", "weights", "best.pt")
    if os.path.exists(best_weights):
        # We move best to the root of the version folder for cleaner structure
        import shutil
        shutil.copy(best_weights, os.path.join(version_dir, "best.pt"))

    # 5. Optional Export
    if config.get('export_onnx'):
        training_logger.info("Exporting to ONNX...")
        model.export(format='onnx', optimize=True, simplify=True)
        # ultralytics drops exported model strictly alongside the weights
        onnx_file = os.path.join(models_dir, f"v{next_version_num}", "weights", "best.onnx")
        if os.path.exists(onnx_file): shutil.copy(onnx_file, os.path.join(version_dir, "best.onnx"))

    if config.get('export_torchscript'):
        training_logger.info("Exporting to TorchScript...")
        model.export(format='torchscript')
        ts_file = os.path.join(models_dir, f"v{next_version_num}", "weights", "best.torchscript")
        if os.path.exists(ts_file): shutil.copy(ts_file, os.path.join(version_dir, "best.torchscript.pt"))
        
    training_logger.info(f"Training Complete. All files stored recursively in: {version_dir}")

if __name__ == "__main__":
    train_yolo()
