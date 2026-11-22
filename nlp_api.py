import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parent
NLP_DIR = PROJECT_ROOT / "models_nlp"

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
    "chilli": ["chilli", "chili", "pepper"],
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

def preprocess_text(text: str) -> str:
    tokens = re.findall(r"[a-z]{2,}", str(text).lower())
    return " ".join(tok for tok in tokens if tok not in CUSTOM_STOPS)

def detect_crop(query: str) -> str | None:
    words = set(re.findall(r"[a-z]+", query.lower()))
    for crop, aliases in CROP_ALIASES.items():
        if words.intersection(aliases):
            return crop
    return None

def suggest_crop_detector(crop: str | None) -> str:
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

app = FastAPI(title="CropGuard AI - NLP Intent & Chat API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "service": "nlp_api",
        "status": "online",
        "categories": list(intent_clf.classes_),
    }

@app.post("/api/nlp/chat")
def classify_and_chat(req: ChatRequest):
    q_tokens = set(re.findall(r"[a-z]+", req.query.lower()))
    q_vec = tfidf_vec.transform([preprocess_text(req.query)])
    probs = intent_clf.predict_proba(q_vec)[0]

    category = str(intent_clf.classes_[probs.argmax()])
    confidence = round(float(probs.max()), 4)

    # Override to Crop Disease when explicit leaf/plant disease symptoms are present
    if category == "Other / General" and q_tokens.intersection(DISEASE_SYMPTOMS):
        category = "Crop Disease"

    routed_section = section_router.get(category, DEFAULT_SECTIONS["Other / General"])
    detected_crop = detect_crop(req.query)
    yolo_model = suggest_crop_detector(detected_crop)

    advisory = ""
    if kb_df is not None and kb_matrix is not None:
        mask = pd.Series(True, index=kb_df.index)

        # Restrict retrieval to the user's crop so chili advice never leaks into rice queries
        if detected_crop:
            aliases = CROP_ALIASES[detected_crop]
            pattern = r"\b(?:" + "|".join(aliases) + r")\b"
            crop_mask = kb_df["search_blob"].str.contains(pattern, regex=True)
            if crop_mask.any():
                mask &= crop_mask
            else:
                mask &= False

        cat_mask = mask & (kb_df["final_category"] == category)
        if cat_mask.any():
            mask = cat_mask

        candidate_idx = np.where(mask.to_numpy())[0]
        if len(candidate_idx) > 0:
            sims = cosine_similarity(q_vec, kb_matrix[candidate_idx]).ravel()
            best_local = int(sims.argmax())
            if sims[best_local] >= 0.25:
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

    return {
        "query": req.query,
        "detected_crop": detected_crop or "general",
        "predicted_category": category,
        "confidence": confidence,
        "relevant_section": routed_section,
        "suggested_yolo_model": yolo_model,
        "response": advisory,
    }