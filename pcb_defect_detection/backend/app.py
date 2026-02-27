import os
import sys
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List
import uvicorn

# Ensure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend.detect import predict_image
from database.db import get_stats
from utils.logger import api_logger

app = FastAPI(title="PCB Defect Detection API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants & Security
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIMES = ["image/jpeg", "image/png", "image/jpg"]

# --- Middlewares ---
@app.middleware("http")
async def validation_middleware(request: Request, call_next):
    """
    Middleware strictly enforcing:
    1. Basic Rate-limiting logging (Extensible to redis/cache limiters later)
    2. Content-Length constraints for file uploads to prevent memory overload on 8GB host
    """
    client_ip = request.client.host
    api_logger.info(f"Incoming request from {client_ip} to {request.url.path}")
    
    # Check explicitly defined Content-Length if present
    if "content-length" in request.headers:
        c_len = int(request.headers.get("content-length"))
        if c_len > MAX_FILE_SIZE:
            api_logger.warning(f"Request rejected due to size. {c_len} bytes.")
            return JSONResponse(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                content={"detail": "Payload strictly exceeds 5MB limit."}
            )
            
    response = await call_next(request)
    return response

# --- Helper Functions ---
async def validate_file(file: UploadFile):
    if file.content_type not in ALLOWED_MIMES:
        api_logger.error(f"Invalid MIME type ({file.content_type}) for file {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, 
            detail="Strictly jpg/jpeg/png files allowed."
        )
        
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        api_logger.error(f"File {file.filename} exceeded 5MB post-stream limit.")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, 
            detail="File size strictly exceeds 5MB limit."
        )
    return contents

# --- Routes ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "service": "PCB Detection Engine"}

@app.get("/stats", status_code=status.HTTP_200_OK)
async def dashboard_stats():
    stats = get_stats()
    return stats

@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict_single(file: UploadFile = File(...)):
    """
    Handles single image prediction. 
    Strict MIME checks and file sizes apply.
    """
    try:
        image_bytes = await validate_file(file)
        prediction = predict_image(image_bytes, file.filename)
        return prediction
    except HTTPException as he:
        # Re-raise explicit HTTP errors generated in validation
        raise he
    except Exception as e:
        api_logger.error(f"Internal processing error on {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}"
        )

@app.post("/predict-batch", status_code=status.HTTP_200_OK)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Handles multiple images natively. Processing happens sequentially on CPU to conserve RAM.
    """
    results = []
    
    if len(files) > 10:
         raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Batch prediction strictly limited to 10 images concurrently."
        )

    for idx, file in enumerate(files):
        try:
            image_bytes = await validate_file(file)
            prediction = predict_image(image_bytes, file.filename)
            results.append({"status": "success", "data": prediction})
        except HTTPException as he:
            results.append({"status": "failed", "filename": file.filename, "error": he.detail})
        except Exception as e:
            results.append({"status": "failed", "filename": file.filename, "error": str(e)})

    return {"batch_results": results}

if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
