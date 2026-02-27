import os
import sys
import time
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend.model_loader import get_inference_model, get_inference_thresholds
from database.db import log_prediction
from utils.logger import inference_logger

# Fallback names if config/dataset structure entirely fails
CLASS_NAMES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}

# Colors config for bounding boxes (BGR format)
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0)   # Cyan
]

def predict_image(image_bytes: bytes, filename: str) -> dict:
    """
    Performs inference on image bytes, draws bounding boxes, logs to DB, 
    and returns a structured JSON-able dictionary.
    """
    inference_logger.info(f"Received inference request for image: {filename}")
    start_time = time.time()
    
    model = get_inference_model()
    thresholds = get_inference_thresholds()
    
    # Pre-process image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        inference_logger.error(f"Failed to decode image: {filename}")
        raise ValueError("Invalid image file.")
        
    # Store original dimensions
    h, w = img.shape[:2]
    
    # Run Inference (device='cpu' heavily enforced implicitly by YOLO initialization + hardware)
    results = model(img, 
                    device='cpu', 
                    conf=thresholds['conf'], 
                    iou=thresholds['iou'], 
                    verbose=False)
                    
    defects = []
    
    # ultralytics results is a list representing batch results. We passed a single image.
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Box coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            
            # Class mapping
            # Attempt to use model names over fallback
            cls_name = model.names[cls_id] if hasattr(model, 'names') and cls_id in model.names else CLASS_NAMES.get(cls_id, f"Unknown_{cls_id}")
            
            defects.append({
                "type": cls_name,
                "confidence": round(conf, 4),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })
            
            # Log to SQLite mapping
            log_prediction(filename, cls_name, conf)
            
            # Draw bounding box
            color = COLORS[cls_id % len(COLORS)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Label background & text
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (int(x1), int(y1) - 20), (int(x1) + tw, int(y1)), color, -1)
            cv2.putText(img, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    # Encode drawn image back to jpeg bytes
    _, encoded_img = cv2.imencode('.jpg', img)
    drawn_bytes_b64 = encoded_img.tobytes()
    
    import base64
    b64_string = base64.b64encode(drawn_bytes_b64).decode('utf-8')
    
    latency = round((time.time() - start_time) * 1000, 2)
    inference_logger.info(f"Inference complete: {len(defects)} defects found. Latency: {latency}ms")
    
    board_status = "Good" if len(defects) == 0 else "Defective"
    routing_decision = "Ready-to-Sell" if len(defects) == 0 else "Rework"
    
    return {
        "filename": filename,
        "inference_time_ms": latency,
        "total_defects": len(defects),
        "board_status": board_status,
        "routing_decision": routing_decision,
        "defects": defects,
        "image_base64": b64_string
    }
