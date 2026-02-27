import os
import sys
import json
import yaml
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.logger import training_logger

def load_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}

def generate_report(version_dir: str):
    """
    Consolidates training, evaluation, benchmark, and config logs into 
    final_report.json and final_report.md
    """
    training_logger.info(f"Generating Final Report for {version_dir}...")
    
    # 1. Gather Data
    config_snap = os.path.join(version_dir, "config_snapshot.yaml")
    eval_met = load_json(os.path.join(version_dir, "evaluation_metrics.json"))
    bench_met = load_json(os.path.join(version_dir, "benchmark.json"))
    
    hash_txt = os.path.join(version_dir, "dataset_hash.txt")
    dataset_hash = "N/A"
    if os.path.exists(hash_txt):
        with open(hash_txt, "r") as f:
            dataset_hash = f.read().strip()
            
    config_data = {}
    if os.path.exists(config_snap):
        with open(config_snap, "r") as f:
            config_data = yaml.safe_load(f)

    # 2. Build JSON
    report = {
        "version": os.path.basename(version_dir),
        "dataset_hash": dataset_hash,
        "parameters": config_data,
        "evaluation": eval_met,
        "benchmark": bench_met,
    }
    
    json_path = os.path.join(version_dir, "final_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # 3. Build Markdown
    md_content = f"""# PCB Defect Detection System - Final Report [{report['version']}]
    
## Dataset Integrity
- **Version Hash (MD5):** `{dataset_hash}`

## Evaluation Metrics
- **mAP50:** {eval_met.get('mAP50', 'N/A')}
- **mAP50-95:** {eval_met.get('mAP50-95', 'N/A')}
- **Precision:** {eval_met.get('precision', 'N/A')}
- **Recall:** {eval_met.get('recall', 'N/A')}

## CPU Benchmark Performance
- **Throughput (img/sec):** {bench_met.get('throughput_images_per_sec', 'N/A')}
- **Avg Inference Latency (ms):** {bench_met.get('avg_inference_time_ms', 'N/A')}
- **Model Fused:** {bench_met.get('model_fused', 'N/A')}
    
## Setup Configuration Snapshot
- **Optimizer:** {config_data.get('optimizer', 'N/A')}
- **LR Scheduler:** {config_data.get('scheduler_type', 'N/A')}
- **Batch Size:** {config_data.get('batch_size', 'N/A')}
- **Epochs:** {config_data.get('epochs', 'N/A')}
- **Deterministic Seed:** {config_data.get('seed', 'N/A')}
    """
    
    md_path = os.path.join(version_dir, "final_report.md")
    with open(md_path, "w") as f:
        f.write(md_content)
        
    training_logger.info(f"Final Report successfully generated at {version_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final_report.py [version_dir]")
        sys.exit(1)
    generate_report(sys.argv[1])
