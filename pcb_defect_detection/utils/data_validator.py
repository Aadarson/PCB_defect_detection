import os
from glob import glob
from utils.logger import system_logger

def validate_yolo_labels(labels_dir: str, num_classes: int) -> tuple:
    """
    Validates YOLO formatted txt files.
    Checks:
    1. Line format: class_id x_center y_center width height
    2. Class id bounds
    3. Bounding box constraints (0-1)
    
    Returns (is_valid: bool, stats: dict)
    """
    label_files = glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    if not label_files:
        system_logger.error("No label files found!")
        return False, {"total": 0, "corrupt": 0}

    corrupt_files = []
    total_boxes = 0

    for file in label_files:
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    total_boxes += 1
                    parts = line.strip().split()
                    
                    if len(parts) != 5:
                        corrupt_files.append((file, "Invalid format (expecting 5 values)"))
                        break
                        
                    cls_id, x, y, w, h = map(float, parts)
                    
                    if not (0 <= cls_id < num_classes):
                        corrupt_files.append((file, f"Class ID out of bounds. Found {cls_id}"))
                        break
                        
                    if not all(0.0 <= val <= 1.0 for val in [x, y, w, h]):
                        corrupt_files.append((file, "Bounding box coordinates outside [0, 1] range"))
                        break
                        
        except Exception as e:
            corrupt_files.append((file, str(e)))

    stats = {
        "total_files": len(label_files),
        "total_boxes": total_boxes,
        "corrupted_files": len(corrupt_files)
    }

    if corrupt_files:
        system_logger.warning(f"Dataset Validation failed on {len(corrupt_files)} files.")
        for f, reason in corrupt_files[:5]:
            system_logger.warning(f"  - {f}: {reason}")
        if len(corrupt_files) > 5:
            system_logger.warning(f"  ... and {len(corrupt_files) - 5} more.")
            
        return False, stats
        
    system_logger.info("Dataset Validation Passed.")
    return True, stats
