import base64
import gc
import io
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_DIR = PROJECT_ROOT / "models_yolo"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

YOLO_WEIGHTS = {
    "cotton": YOLO_DIR / "best_CottonDisease_Clean.pt",
    "rice": YOLO_DIR / "best_RiceDisease_Clean.pt",
    "multicrop": YOLO_DIR / "best_MultiCrop_Clean.pt",
    "plantdoc": YOLO_DIR / "best_PlantDoc_Clean.pt",
    "unified": YOLO_DIR / "best.pt",
}

AVAILABLE_MODELS = {k: p for k, p in YOLO_WEIGHTS.items() if p.exists()}
_cache = {"key": None, "model": None}


def get_yolo_model(key: str) -> YOLO:
    if _cache["key"] != key:
        _cache["model"] = None
        gc.collect()
        _cache["model"] = YOLO(str(AVAILABLE_MODELS[key]))
        _cache["key"] = key
    return _cache["model"]


def resolve_crop_model(crop_hint: str, filename: str = "") -> str:
    for text in (crop_hint, filename):
        h = (text or "").lower().strip()
        if h in AVAILABLE_MODELS:
            return h
        if any(k in h for k in ["rice", "paddy"]):
            return "rice"
        if "cotton" in h:
            return "cotton"
        if any(k in h for k in ["banana", "cauliflower", "chilli", "chili", "pepper", "groundnut", "peanut", "radish"]):
            return "multicrop"
    if "unified" in AVAILABLE_MODELS and (crop_hint or "").lower().strip() in {"auto", "all"}:
        return "unified"
    return "plantdoc"


app = FastAPI(title="CropGuard AI - YOLOv8 Leaf Disease API", version="1.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "service": "dl_api",
        "status": "online",
        "device": str(DEVICE),
        "available_models": list(AVAILABLE_MODELS.keys()),
    }


@app.post("/api/dl/detect")
async def detect_leaf_disease(
    file: UploadFile = File(...),
    crop: Optional[str] = Form("auto"),
    conf: Optional[float] = Form(0.35),
):
    model_key = resolve_crop_model(crop or "auto", file.filename or "")
    if model_key not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail=f"Model '{model_key}' is not available.")

    try:
        img_bytes = await file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    model = get_yolo_model(model_key)
    used_conf = float(conf) if conf is not None else 0.35

    res = model.predict(
        source=pil_img,
        conf=used_conf,
        imgsz=640,
        device=DEVICE,
        verbose=False,
    )[0]

    # Retry once at a lower threshold if no detections are found on the first pass
    if len(res.boxes) == 0 and used_conf > 0.15:
        used_conf = 0.15
        res = model.predict(
            source=pil_img,
            conf=used_conf,
            imgsz=640,
            device=DEVICE,
            verbose=False,
        )[0]

    detections = [
        {
            "disease_class": res.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 4),
            "bbox_xyxy": [round(float(v), 1) for v in box.xyxy[0].tolist()],
        }
        for box in res.boxes
    ]

    annotated_pil = Image.fromarray(res.plot()[:, :, ::-1])
    buf = io.BytesIO()
    annotated_pil.save(buf, format="JPEG", quality=85)
    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "model_used": model_key,
        "confidence_threshold_used": used_conf,
        "total_detections": len(detections),
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{b64_img}",
    }