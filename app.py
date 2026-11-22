import re
from pathlib import Path
import joblib
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from ultralytics import YOLO

PROJECT_ROOT = Path(r"C:\Users\91934\OneDrive\Desktop\CropGuard_AI-DL")
YOLO_DIR = PROJECT_ROOT / "models_yolo"
NLP_DIR = PROJECT_ROOT / "models_nlp"

YOLO_MODELS = {
    "Rice (3 classes)": YOLO_DIR / "best_RiceDisease_Clean.pt",
    "Cotton (5 classes)": YOLO_DIR / "best_CottonDisease_Clean.pt",
    "Multi-Crop: Banana, Cauliflower, Chilli, Groundnut, Radish (30 classes)": YOLO_DIR / "best_MultiCrop_Clean.pt",
    "PlantDoc General Crops (29 classes)": YOLO_DIR / "best_PlantDoc_Clean.pt",
}

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

def preprocess_text(text: str) -> str:
    tokens = re.findall(r"[a-z]{2,}", str(text).lower())
    return " ".join(tok for tok in tokens if tok not in CUSTOM_STOPS)

@st.cache_resource
def load_nlp_artifacts():
    vec = joblib.load(NLP_DIR / "tfidf_vectorizer.pkl")
    clf = joblib.load(NLP_DIR / "farmerchat_intent_classifier.pkl")
    router_path = NLP_DIR / "cropguard_section_router.pkl"
    router = joblib.load(router_path) if router_path.exists() else DEFAULT_SECTIONS
    return vec, clf, router

@st.cache_resource
def load_yolo_model(model_path: str):
    return YOLO(model_path)

def auto_select_detector(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["rice", "paddy"]):
        return "Rice (3 classes)"
    if "cotton" in q:
        return "Cotton (5 classes)"
    if any(k in q for k in ["banana", "cauliflower", "chilli", "chili", "groundnut", "radish"]):
        return "Multi-Crop: Banana, Cauliflower, Chilli, Groundnut, Radish (30 classes)"
    return "PlantDoc General Crops (29 classes)"

st.set_page_config(page_title="CropGuard AI", layout="wide")
st.title("CropGuard AI — Intelligent Crop Disease & Farmer Query Assistant")

vec, clf, router = load_nlp_artifacts()

st.subheader("1. Farmer Query Intent Classification (NLP)")
query = st.text_input(
    "Enter a farmer question:",
    value="My rice leaves are turning yellow."
)

suggested_detector = "Rice (3 classes)"
if query.strip():
    probs = clf.predict_proba(vec.transform([preprocess_text(query)]))[0]
    pred_category = clf.classes_[probs.argmax()]
    confidence = float(probs.max())
    target_section = router.get(pred_category, DEFAULT_SECTIONS["Other / General"])
    suggested_detector = auto_select_detector(query)

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Category", pred_category)
    c2.metric("Confidence", f"{confidence:.1%}")
    c3.info(f"**Routed Section:** {target_section}")

st.divider()
st.subheader("2. Leaf Disease Object Detection (YOLOv8)")

col_cfg, col_img = st.columns([1, 2])
with col_cfg:
    detector_keys = list(YOLO_MODELS.keys())
    selected_key = st.selectbox(
        "Select YOLOv8 Crop Detector",
        detector_keys,
        index=detector_keys.index(suggested_detector)
    )
    conf_thresh = st.slider("Confidence Threshold", 0.10, 0.80, 0.25, 0.05)
    uploaded = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png", "webp"])

with col_img:
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        yolo = load_yolo_model(str(YOLO_MODELS[selected_key]))
        res = yolo.predict(source=pil_img, conf=conf_thresh, verbose=False)[0]
        annotated = Image.fromarray(res.plot()[:, :, ::-1])
        st.image(annotated, caption=f"YOLOv8 Predictions ({selected_key})", use_container_width=True)

        if len(res.boxes) == 0:
            st.warning(f"No detections above {conf_thresh:.2f}. Try lowering the confidence threshold.")
        else:
            for box in res.boxes:
                cls_name = res.names[int(box.cls[0])]
                score = float(box.conf[0])
                st.success(f"Detected: **{cls_name}** ({score:.1%} confidence)")
