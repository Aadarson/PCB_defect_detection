import os
from glob import glob
from collections import Counter
import hashlib
from utils.logger import system_logger

def analyze_class_distribution(labels_dir: str, num_classes: int, threshold: float = 0.4) -> dict:
    """
    Parses all validation/train YOLO label files and counts occurrences of each class.
    Returns weights to handle imbalance.
    """
    label_files = glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    class_counts = Counter()

    for file in label_files:
        with open(file, 'r') as f:
            for line in f.readlines():
                try:
                    cls_id = int(float(line.strip().split()[0]))
                    class_counts[cls_id] += 1
                except:
                    continue

    total_instances = sum(class_counts.values())
    
    if total_instances == 0:
        system_logger.error("No instances found during class distribution analysis.")
        return {"weights": [1.0] * num_classes, "imbalance_detected": False}

    # Detect if any single class dominates > threshold% of the dataset
    imbalance_detected = any(count/total_instances > threshold for count in class_counts.values())
    
    if imbalance_detected:
        system_logger.warning("Class imbalance detected in dataset!")
    
    # Calculate inverse frequency weights
    # Weight = total / (num_classes * count)
    weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        if count == 0:
            system_logger.warning(f"Class {cls} has 0 instances!")
            weights.append(1.0) # Fallback
        else:
            w = total_instances / (num_classes * count)
            weights.append(round(w, 4))

    system_logger.info(f"Class Distribution: {dict(class_counts)}")
    system_logger.info(f"Calculated Weights: {weights}")
    
    return {
        "counts": dict(class_counts),
        "weights": weights,
        "imbalance_detected": imbalance_detected,
        "total_instances": total_instances
    }

def generate_dataset_hash(labels_dir: str) -> str:
    """
    Generates an MD5 hash representing the current dataset state based on all label files.
    Ensures absolute reproducibility mapping.
    """
    system_logger.info("Generating dataset MD5 hash for version tracking...")
    hash_md5 = hashlib.md5()
    
    label_files = sorted(glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True))
    
    for file in label_files:
        with open(file, "rb") as f:
            # Read in chunks to avoid memory issues on huge datasets
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
                
    final_hash = hash_md5.hexdigest()
    system_logger.info(f"Dataset Hash: {final_hash}")
    return final_hash
