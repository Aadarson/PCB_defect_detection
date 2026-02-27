import os
import sys
import time
import json
import psutil
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import training_logger

def run_benchmark(version_dir: str, data_yaml: str):
    """
    Runs CPU throughput benchmark for the trained model.
    Saves metrics into benchmark.json inside the version folder.
    """
    model_path = os.path.join(version_dir, "best.pt")
    if not os.path.exists(model_path):
        training_logger.error(f"Cannot benchmark: {model_path} does not exist.")
        return
        
    training_logger.info(f"Benchmarking CPU Throughput for {model_path}")
    
    # Use ultralytics built-in benchmark utility
    # We will wrap it or extract times
    model = YOLO(model_path)
    model.fuse() # Crucial for CPU optimization
    
    # We simulate an inference loop to measure average throughput (ms/image)
    # Finding a few evaluation images
    import glob
    dataset_dir = os.path.dirname(data_yaml)
    val_images = glob.glob(os.path.join(dataset_dir, "images", "val", "*.jpg"))
    if not val_images:
        val_images = glob.glob(os.path.join(dataset_dir, "images", "val", "*.png"))
        
    if not val_images:
        training_logger.warning("No validation images found to benchmark against.")
        return
        
    test_images = val_images[:min(50, len(val_images))] # Test on max 50 images
    
    start_time = time.time()
    for img in test_images:
        _ = model(img, device='cpu', verbose=False)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / len(test_images)) * 1000
    images_per_sec = len(test_images) / total_time
    mem_usage = psutil.virtual_memory().percent
    
    benchmark_data = {
        "device": "cpu",
        "model_fused": True,
        "images_tested": len(test_images),
        "total_time_seconds": round(total_time, 2),
        "avg_inference_time_ms": round(avg_inference_time_ms, 2),
        "throughput_images_per_sec": round(images_per_sec, 2),
        "memory_usage_snapshot_percent": mem_usage
    }
    
    dest = os.path.join(version_dir, "benchmark.json")
    with open(dest, "w") as f:
        json.dump(benchmark_data, f, indent=4)
        
    training_logger.info(f"Benchmark complete. {images_per_sec:.2f} img/s. Saved to {dest}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py [version_dir] [data_yaml_path]")
        sys.exit(1)
    run_benchmark(sys.argv[1], sys.argv[2])
