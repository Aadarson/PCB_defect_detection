# PCB Defect Detection System (YOLOv8)

A production-ready, research-grade deep learning system for detecting PCB manufacturing defects. Optimized exclusively for CPU inference and training on limited hardware (8GB RAM Windows 11 host).

## System Architecture

> **Frontend**: HTML5/VanillaJS/CSS3 Dark Theme UI
> **Backend API**: FastAPI (Uvicorn, REST Endpoints, Strict MIME/Size checking)
> **Database**: SQLite3 Local file logging (Predictions & History)
> **Model Loader**: Fused YOLOv8 Nano weights instantiated on Native CPU threads
> **Training Pipeline**: Fully deterministic, reproducible Ultralytics YOLO logic mapped to a YAML config.

---

## 🚀 Deployment Guide

### 1. Pre-flight Checks
Before starting any servers, verify your dataset and configuration boundaries:
```bash
python utils/system_check.py
```

### 2. Run Backend
The backend utilizes FastAPI. We supply a Dockerfile but you can run it purely natively:
```bash
pip install -r requirements.txt
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

### 3. Open Frontend
Simply open `frontend/index.html` in any modern web browser or host via Live Server. No build step is required.

---

## 🧠 Training Workflow

### Configuration
All training parameters are abstracted out of code into `training/config.yaml`. This controls deterministic seeds, learning rate schedulers, image augmentation, memory limits, and more.

### Start Training
```bash
python training/train.py
```
> **Memory Guard Active**: If your host RAM usage strictly exceeds `85%`, training will gracefully generate a checkpoint and halt to safely prevent your 8GB RAM Windows system from Bluescreening due to OOM errors.

---

## 📈 Model Card: PCB Defect Detector v1

### Overview
- **Architecture**: YOLOv8 Nano (`yolov8n.pt`)
- **Purpose**: High-speed, low-resource defect inspection for Printed Circuit Boards.
- **Target Hardware**: Intel/AMD x86 CPUs with ≥ 8GB RAM.
- **Intended Use**: Industrial batch inspection, Academic ML research, demonstrations.

### Model Limitations
1. **Lighting Constraints**: Extremely sensitive to non-uniform, overly bright, or overly shadowed macro photography.
2. **Alignment Considerations**: Works best directly top-down. Severe angular distortion impairs bounding box confidence.
3. **Hardware Barrier**: Training times on pure CPU are notoriously slow. Benchmark throughputs are optimized up to ~5-15 images/sec depending on host clock speeds.

### Ethical & Bias Considerations
- **Manufacturing Bias**: Models trained on single-factory datasets will underperform in zero-shot cross-factory transfer.
- **Safety Critical Risks**: False negatives (missing a critical defect) may result in severe downstream deployment hazards in embedded hardware. **Human-in-the-loop review is mandatory for mission-critical paths.**

### Overfitting Mitigation
We actively employ:
- **Cosine Annealing LR Schedulers** to dynamically adjust descent.
- **Auto-Monitoring Hooks** evaluating `val_loss` divergence over 3 consecutive epochs.

### Reproducibility
- Data undergoes SHA256 integrity hashing into `dataset_hash.txt`.
- `torch.use_deterministic_algorithms()` and global `random.seed(42)` flags are explicitly applied.

---

## ✅ Final Architecture Validation Checklist
- [x] Dataset structure mapping (YOLO standard format enabled)
- [x] Config parameterization mapped snapshot creation
- [x] Benchmarking layer enabled
- [x] Memory limit circuit breaker (psutil) mapped
- [x] Dataset Class Imbalance analysis applied
- [x] FastAPI MIME injection and Rate-Limit middleware applied
- [x] Reproducibility logic confirmed

## Future Improvements
- **Edge Deployment**: Compiling to TensorRT/OpenVINO architectures.
- **Real-Time Feed**: Supporting streaming endpoints (`/video_feed`) directly from mounted webcams.
