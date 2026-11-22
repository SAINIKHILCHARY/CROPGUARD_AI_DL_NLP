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

SPECIFIC_DISEASE_GUIDANCE = {
    "brown spot": (
        "Rice Brown Spot (Bipolaris oryzae) Treatment & Prevention:\n"
        "• Immediate Action: Pause nitrogen (urea) top-dressing and drain excess stagnant water.\n"
        "• Nutrient Correction: Brown Spot is strongly linked to potassium deficiency—apply Muriate of Potash (MOP) if the crop is in the vegetative or tillering stage.\n"
        "• Targeted Fungicide: Spray Mancozeb 75% WP (2.5 g/L of water) or Propiconazole 25% EC (1 ml/L) at early symptom onset; repeat after 10–14 days if needed.\n"
        "• Prevention: Use certified seed treated with Pseudomonas fluorescens (10 g/kg) or Carbendazim + Mancozeb (2 g/kg) next season, and destroy infected stubble."
    ),
    "bacterial leaf blight": (
        "Rice Bacterial Leaf Blight Treatment & Prevention:\n"
        "• Immediate Action: Drain standing water from the field immediately and stop nitrogen fertilizer application.\n"
        "• Nutrient Correction: Apply potash (MOP) to strengthen leaf tissue resistance.\n"
        "• Targeted Control: Avoid fungal sprays (ineffective against bacteria). Apply Copper Hydroxide 77% WP (2 g/L) or Copper Oxychloride 50% WP (2.5 g/L) on early-infected patches.\n"
        "• Prevention: Avoid clipping seedling tips during transplanting and keep field bunds free of weed hosts."
    ),
    "leaf spot": (
        "Leaf Spot Treatment & Prevention:\n"
        "• Immediate Action: Prune and destroy heavily spotted lower leaves to reduce spore spread and avoid overhead irrigation.\n"
        "• Targeted Treatment: Spray Mancozeb 75% WP (2.5 g/L of water) or Carbendazim 50% WP (1 g/L) for fungal spots; if water-soaked bacterial lesions appear, use Copper Oxychloride 50% WP (2.5–3 g/L).\n"
        "• Prevention: Maintain proper plant spacing for airflow, use disease-free seeds, and follow crop rotation."
    ),
    "downy mildew": (
        "Downy Mildew Treatment & Prevention:\n"
        "• Immediate Action: Remove infected outer leaves and avoid late-evening overhead watering.\n"
        "• Targeted Fungicide: Spray Metalaxyl 8% + Mancozeb 64% WP (2 g/L of water) or Cymoxanil + Mancozeb at early yellowing.\n"
        "• Prevention: Ensure well-drained raised beds and follow a 2–3 year non-host crop rotation."
    ),
    "early rust": (
        "Rust Disease Treatment & Prevention:\n"
        "• Immediate Action: Inspect lower leaves early for orange-brown pustules and avoid excessive nitrogen.\n"
        "• Targeted Fungicide: Spray Hexaconazole 5% EC (2 ml/L of water), Propiconazole 25% EC (1 ml/L), or Chlorothalonil 75% WP (2 g/L).\n"
        "• Prevention: Remove volunteer plants after harvest and rotate with cereal crops."
    ),
}

CROP_DISEASE_ADVISORIES = {
    "rice": (
        "Your Rice symptoms indicate a possible leaf disease (such as Brown Spot or Leaf Blight).\n"
        "• Immediate Action: Pause urea/nitrogen top-dressing, drain stagnant water, and apply balanced Muriate of Potash (MOP) if soil potassium is low.\n"
        "• Scanner Confirmation: Upload a clear close-up photo of the affected leaf to the 'rice' YOLOv8 scanner to distinguish fungal Brown Spot (treat with Mancozeb 2.5 g/L or Propiconazole 1 ml/L) from Bacterial Leaf Blight (manage water & copper bactericide).\n"
        "• Prevention: Treat seeds with Pseudomonas fluorescens (10 g/kg) or Carbendazim (2 g/kg) before sowing."
    ),
    "cotton": (
        "Your Cotton symptoms indicate a possible leaf disease (such as Bacterial Blight or Alternaria Leaf Spot).\n"
        "• Immediate Action: Remove severely infected lower leaves and avoid overhead irrigation.\n"
        "• Scanner Confirmation: Upload a leaf photo to the 'cotton' YOLOv8 scanner. For fungal spots, spray Mancozeb 75% WP (2.5 g/L); for Bacterial Blight, spray Copper Oxychloride 50% WP (3 g/L).\n"
        "• Prevention: Practice crop rotation, destroy post-harvest cotton stalks, and maintain balanced NPK nutrition."
    ),
    "chilli": (
        "Your Chilli symptoms indicate a possible leaf disease (such as Cercospora Leaf Spot or Leaf Curl).\n"
        "• Immediate Action: Remove severely affected leaves/plants early and control whiteflies or thrips using yellow/blue sticky traps.\n"
        "• Scanner Confirmation: Upload a close-up leaf photo to the 'multicrop' YOLOv8 scanner. For fungal spots, spray Mancozeb (2.5 g/L) or Copper Oxychloride (2.5 g/L).\n"
        "• Prevention: Use virus-free seedlings and keep field borders weed-free."
    ),
    "banana": (
        "Your Banana symptoms indicate a possible leaf disease (such as Sigatoka or Cordana Leaf Spot).\n"
        "• Immediate Action: Prune and destroy heavily yellowed/spotted older leaves to reduce spore load.\n"
        "• Scanner Confirmation: Upload a close-up leaf photo to the 'multicrop' YOLOv8 scanner. Spray Propiconazole 25% EC (1 ml/L) or Mancozeb 75% WP (2.5 g/L) with a sticker/wetting agent.\n"
        "• Prevention: Ensure good field drainage, proper spacing, and adequate potassium fertilization."
    ),
    "cauliflower": (
        "Your Cauliflower symptoms indicate a possible leaf disease (such as Alternaria Leaf Spot or Downy Mildew).\n"
        "• Immediate Action: Remove infected outer leaves and avoid overhead watering late in the day.\n"
        "• Scanner Confirmation: Upload a close-up leaf photo to the 'multicrop' YOLOv8 scanner. Spray Mancozeb 75% WP (2 g/L) or Metalaxyl + Mancozeb (2 g/L).\n"
        "• Prevention: Follow a 2–3 year non-brassica crop rotation and use treated seeds."
    ),
    "groundnut": (
        "Your Groundnut symptoms indicate a possible leaf disease (such as Tikka Leaf Spot or Rust).\n"
        "• Immediate Action: Monitor lower canopy leaves at 30–35 days after sowing.\n"
        "• Scanner Confirmation: Upload a close-up leaf photo to the 'multicrop' YOLOv8 scanner. Spray Chlorothalonil 75% WP (2 g/L) or Hexaconazole 5% EC (2 ml/L).\n"
        "• Prevention: Remove volunteer groundnut plants and rotate with cereals."
    ),
    "radish": (
        "Your Radish symptoms indicate a possible leaf disease (such as Alternaria Blight or White Rust).\n"
        "• Immediate Action: Thin dense rows to improve airflow and reduce leaf wetness.\n"
        "• Scanner Confirmation: Upload a close-up leaf photo to the 'multicrop' YOLOv8 scanner. Spray Mancozeb 75% WP (2 g/L) respecting pre-harvest intervals.\n"
        "• Prevention: Sow clean seed on well-drained beds and rotate crops."
    ),
}


def preprocess_text(text: str) -> str:
    tokens = re.findall(r"[a-z]{2,}", str(text).lower())
    return " ".join(tok for tok in tokens if tok not in CUSTOM_STOPS)


def detect_crop(query: str) -> str | None:
    normalized = query.lower().replace("_", " ")
    words = set(re.findall(r"[a-z]+", normalized))
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


def build_disease_advisory(query: str, detected_crop: str | None, yolo_model: str) -> str:
    q_norm = query.lower().replace("_", " ")
    for disease_key, guidance in SPECIFIC_DISEASE_GUIDANCE.items():
        if disease_key in q_norm:
            return guidance

    if detected_crop and detected_crop in CROP_DISEASE_ADVISORIES:
        return CROP_DISEASE_ADVISORIES[detected_crop]

    crop_label = detected_crop.capitalize() if detected_crop else "crop"
    return (
        f"Your {crop_label} symptoms indicate a possible leaf disease (such as Brown Spot or Leaf Blight). "
        f"Upload a clear close-up photo of the affected leaf to the '{yolo_model}' YOLOv8 scanner for exact "
        "diagnosis and targeted fungicide/nutrient treatment."
    )


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

app = FastAPI(title="CropGuard AI - NLP Intent & Chat API", version="1.2")

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
    normalized_query = req.query.replace("_", " ")
    q_tokens = set(re.findall(r"[a-z]+", normalized_query.lower()))
    q_vec = tfidf_vec.transform([preprocess_text(normalized_query)])
    probs = intent_clf.predict_proba(q_vec)[0]

    category = str(intent_clf.classes_[probs.argmax()])
    confidence = round(float(probs.max()), 4)

    # Override to Crop Disease when explicit leaf/plant disease symptoms are present
    if category == "Other / General" and q_tokens.intersection(DISEASE_SYMPTOMS):
        category = "Crop Disease"

    routed_section = section_router.get(category, DEFAULT_SECTIONS["Other / General"])
    detected_crop = detect_crop(normalized_query)
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
        if category == "Crop Disease":
            advisory = build_disease_advisory(normalized_query, detected_crop, yolo_model)
        else:
            crop_label = detected_crop.capitalize() if detected_crop else "crop"
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