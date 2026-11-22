import json
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
NLP_DIR = PROJECT_ROOT / "models_nlp"
STORE_DIR = PROJECT_ROOT / "runtime_store"
STORE_DIR.mkdir(parents=True, exist_ok=True)

CHAT_LOG_PATH = STORE_DIR / "chat_history.jsonl"
DETECTION_LOG_PATH = STORE_DIR / "detection_history.jsonl"
_store_lock = threading.Lock()

KB_CSV = NLP_DIR / "farmerchat_v7_reviewed.csv"
if not KB_CSV.exists():
    KB_CSV = PROJECT_ROOT / "raw" / "05_FarmerChat_NLP" / "farmerchat_v7_reviewed.csv"

DEFAULT_SECTIONS = {
    "Crop Disease": "Leaf Disease Detector (YOLOv8 Scanner & Treatment Advisory)",
    "Irrigation": "Smart Irrigation & Water Management Scheduler",
    "Fertilizer": "Soil Nutrition & Fertilizer Dosage Advisor",
    "Crop Recommendation": "Telangana Crop & Variety Selection Engine",
    "Pest Management": "Integrated Pest & Weed Control Guide",
    "Crop Management": "Sowing, Cultivation & Harvest Planner",
    "Other / General": "General Agri-Helpline & Mandi Market Desk",
}

KEEP_WORDS = {"how", "much", "which", "when"}
CUSTOM_STOPS = (set(ENGLISH_STOP_WORDS) - KEEP_WORDS) | {"telangana", "india", "hyderabad"}

CROP_ALIASES = {
    "rice": ["rice", "paddy"],
    "cotton": ["cotton"],
    "chilli": ["chilli", "chili", "pepper", "capsicum"],
    "banana": ["banana"],
    "groundnut": ["groundnut", "peanut"],
    "cauliflower": ["cauliflower"],
    "radish": ["radish"],
    "maize": ["maize", "corn"],
    "tomato": ["tomato"],
    "potato": ["potato"],
}

DISEASE_SYMPTOMS = {
    "spot", "spots", "blight", "rust", "mildew", "wilt", "rot",
    "curl", "lesion", "lesions", "mosaic", "fungus", "fungal",
    "bacterial", "disease", "infected", "yellowing", "blast",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp_path.replace(path)


def append_chat_record(record: Dict[str, Any]) -> None:
    with _store_lock:
        with CHAT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_recent_chats(session_id: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
    with _store_lock:
        items = _read_jsonl(CHAT_LOG_PATH)
    if session_id:
        items = [x for x in items if x.get("session_id") == session_id]
    return items[-limit:]


def upsert_detection_record(record: Dict[str, Any]) -> Dict[str, Any]:
    with _store_lock:
        items = _read_jsonl(DETECTION_LOG_PATH)
        for idx, existing in enumerate(items):
            if existing.get("id") == record["id"]:
                merged = {**existing, **record, "updated_at": utc_now()}
                items[idx] = merged
                _write_jsonl(DETECTION_LOG_PATH, items)
                return merged
        record.setdefault("created_at", utc_now())
        record["updated_at"] = record["created_at"]
        items.append(record)
        _write_jsonl(DETECTION_LOG_PATH, items)
        return record


def list_detection_records(session_id: Optional[str] = None, limit: int = 25) -> List[Dict[str, Any]]:
    with _store_lock:
        items = _read_jsonl(DETECTION_LOG_PATH)
    if session_id:
        items = [x for x in items if x.get("session_id") == session_id]
    return list(reversed(items[-limit:]))


def preprocess_text(text: str) -> str:
    tokens = re.findall(r"[a-z]{2,}", str(text).lower())
    return " ".join(tok for tok in tokens if tok not in CUSTOM_STOPS)


def detect_crop(query: str) -> Optional[str]:
    normalized = query.lower().replace("_", " ")
    words = set(re.findall(r"[a-z]+", normalized))
    for crop, aliases in CROP_ALIASES.items():
        if words.intersection(aliases):
            return crop
    return None


def suggest_crop_detector(crop: Optional[str]) -> str:
    if crop == "rice":
        return "rice"
    if crop == "cotton":
        return "cotton"
    if crop in {"banana", "cauliflower", "chilli", "groundnut", "radish"}:
        return "multicrop"
    return "plantdoc"


def clean_advisory_text(text: str) -> str:
    cleaned = re.sub(r"^Hello!\s*\[XXX\][.,!]?\s*", "", text.strip(), flags=re.IGNORECASE)
    return cleaned.replace("[XXX]", "Farmer").strip()


tfidf_vec = joblib.load(NLP_DIR / "tfidf_vectorizer.pkl")
intent_clf = joblib.load(NLP_DIR / "farmerchat_intent_classifier.pkl")
router_path = NLP_DIR / "cropguard_section_router.pkl"
section_router = joblib.load(router_path) if router_path.exists() else DEFAULT_SECTIONS

kb_df, kb_matrix = None, None
if KB_CSV.exists():
    raw_df = pd.read_csv(KB_CSV)
    if {"query", "response", "final_category"}.issubset(raw_df.columns):
        kb_df = (
            raw_df[["query", "response", "final_category"]]
            .dropna()
            .drop_duplicates(subset=["query"])
            .reset_index(drop=True)
        )
        kb_df["search_blob"] = (kb_df["query"] + " " + kb_df["response"]).str.lower()
        kb_matrix = tfidf_vec.transform(kb_df["query"].apply(preprocess_text))

app = FastAPI(title="CropGuard AI - NLP, History & Learning API", version="1.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    search_history: Optional[List[str]] = None


class DetectionRecordInput(BaseModel):
    id: Optional[str] = None
    session_id: Optional[str] = "default"
    filename: Optional[str] = None
    selected_crop: Optional[str] = "auto"
    model_used: str
    confidence_threshold_used: float = 0.35
    total_detections: int = 0
    detections: List[Dict[str, Any]] = []
    notes: Optional[str] = None
    verified_disease: Optional[str] = None


class DetectionUpdateInput(BaseModel):
    selected_crop: Optional[str] = None
    verified_disease: Optional[str] = None
    notes: Optional[str] = None
    detections: Optional[List[Dict[str, Any]]] = None


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "service": "nlp_api",
        "status": "online",
        "categories": list(intent_clf.classes_),
        "stored_chats": len(_read_jsonl(CHAT_LOG_PATH)),
        "stored_detections": len(_read_jsonl(DETECTION_LOG_PATH)),
    }


@app.post("/api/nlp/chat")
def classify_and_chat(req: ChatRequest):
    session_id = req.session_id or "default"
    past_chats = get_recent_chats(session_id=session_id, limit=6)
    past_scans = list_detection_records(session_id=session_id, limit=5)

    # Build enriched context from current query + recent chat history + past searches/scans
    context_snippets: List[str] = [req.query]
    for item in past_chats:
        if item.get("query"):
            context_snippets.append(item["query"])
    for term in (req.search_history or [])[-6:]:
        if term:
            context_snippets.append(term)
    for scan in past_scans:
        for det in scan.get("detections", []):
            if det.get("disease_class"):
                context_snippets.append(str(det["disease_class"]).replace("_", " "))

    enriched_text = " ".join(context_snippets).replace("_", " ")
    q_tokens = set(re.findall(r"[a-z]+", req.query.lower().replace("_", " ")))
    q_vec = tfidf_vec.transform([preprocess_text(req.query)])
    ctx_vec = tfidf_vec.transform([preprocess_text(enriched_text)])

    probs = intent_clf.predict_proba(q_vec)[0]
    category = str(intent_clf.classes_[probs.argmax()])
    confidence = round(float(probs.max()), 4)

    if category == "Other / General" and q_tokens.intersection(DISEASE_SYMPTOMS):
        category = "Crop Disease"

    routed_section = section_router.get(category, DEFAULT_SECTIONS["Other / General"])

    # Prefer crop in current query; fall back to most recent chat/scan crop if omitted
    detected_crop = detect_crop(req.query)
    if not detected_crop:
        for past in reversed(past_chats):
            if past.get("detected_crop") and past["detected_crop"] != "general":
                detected_crop = past["detected_crop"]
                break
    if not detected_crop and past_scans:
        scan_crop = (past_scans[0].get("selected_crop") or "").lower()
        if scan_crop in CROP_ALIASES:
            detected_crop = scan_crop

    yolo_model = suggest_crop_detector(detected_crop)

    advisory = ""
    if kb_df is not None and kb_matrix is not None:
        mask = pd.Series(True, index=kb_df.index)
        if detected_crop:
            aliases = CROP_ALIASES[detected_crop]
            pattern = r"\b(?:" + "|".join(aliases) + r")\b"
            crop_mask = kb_df["search_blob"].str.contains(pattern, regex=True)
            if crop_mask.any():
                mask &= crop_mask

        cat_mask = mask & (kb_df["final_category"] == category)
        if cat_mask.any():
            mask = cat_mask

        candidate_idx = np.where(mask.to_numpy())[0]
        if len(candidate_idx) > 0:
            # Blend current query similarity (0.7) with historical chat/search context (0.3)
            q_sims = cosine_similarity(q_vec, kb_matrix[candidate_idx]).ravel()
            c_sims = cosine_similarity(ctx_vec, kb_matrix[candidate_idx]).ravel()
            blended_sims = 0.7 * q_sims + 0.3 * c_sims
            best_local = int(blended_sims.argmax())
            if blended_sims[best_local] >= 0.20:
                advisory = clean_advisory_text(str(kb_df.loc[candidate_idx[best_local], "response"]))

    if not advisory:
        crop_label = detected_crop.capitalize() if detected_crop else "crop"
        if category == "Crop Disease":
            advisory = (
                f"Your {crop_label} symptoms indicate a possible leaf disease (such as Brown Spot or Leaf Blight). "
                f"Upload a clear close-up photo of the affected leaf to the '{yolo_model}' YOLOv8 scanner for exact "
                "diagnosis and targeted fungicide/nutrient treatment."
            )
        else:
            advisory = (
                f"Routed to {routed_section} for {crop_label}. "
                "Provide crop variety, growth stage, or upload a leaf image for tailored guidance."
            )

    chat_entry = {
        "id": str(uuid.uuid4()),
        "session_id": session_id,
        "timestamp": utc_now(),
        "query": req.query,
        "detected_crop": detected_crop or "general",
        "predicted_category": category,
        "confidence": confidence,
        "relevant_section": routed_section,
        "suggested_yolo_model": yolo_model,
        "response": advisory,
    }
    append_chat_record(chat_entry)

    return {
        **chat_entry,
        "history_context_used": len(past_chats) + len(past_scans),
    }


@app.get("/api/history/chats")
def get_chat_history(session_id: Optional[str] = None, limit: int = 25):
    return {"items": get_recent_chats(session_id=session_id, limit=limit)}


@app.post("/api/history/detections")
def create_or_save_detection(payload: DetectionRecordInput):
    record = payload.model_dump()
    record["id"] = payload.id or str(uuid.uuid4())
    record["session_id"] = payload.session_id or "default"
    saved = upsert_detection_record(record)
    return {"item": saved}


@app.get("/api/history/detections")
def get_detections(session_id: Optional[str] = None, limit: int = 25):
    return {"items": list_detection_records(session_id=session_id, limit=limit)}


@app.patch("/api/history/detections/{scan_id}")
def update_detection(scan_id: str, updates: DetectionUpdateInput):
    with _store_lock:
        items = _read_jsonl(DETECTION_LOG_PATH)
        for idx, item in enumerate(items):
            if item.get("id") == scan_id:
                patch = {k: v for k, v in updates.model_dump().items() if v is not None}
                if "verified_disease" in patch and patch["verified_disease"]:
                    patch["user_verified"] = True
                items[idx] = {**item, **patch, "updated_at": utc_now()}
                _write_jsonl(DETECTION_LOG_PATH, items)
                return {"item": items[idx]}
    raise HTTPException(status_code=404, detail=f"Detection record '{scan_id}' not found.")