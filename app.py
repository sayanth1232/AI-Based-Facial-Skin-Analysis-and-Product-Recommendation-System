import cv2
import streamlit as st
import torch
import timm
import sqlite3
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as T
from ingredient_recommender import load_ingredients, recommend_ingredients
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from collections import Counter


# ---------------- GPT-2 MODEL ----------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


# ---------------- GPT-2 Routine Generator ----------------
def generate_ai_routine_gpt2(skin_type, concern, products):
    """Generates a personalized skincare routine using GPT-2."""
    prompt = f"""
    The user has {skin_type} skin and concerns: {concern}.
    Recommend a daily skincare routine using these ingredients or products: {', '.join(products)}.
    Explain what to use in the morning and at night in 4–6 friendly sentences.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    routine = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if routine.startswith(prompt.strip()):
        routine = routine[len(prompt.strip()):].strip()
    return routine


# ---------------- CONFIG ----------------
VIT_MODEL_PATH = "models/vit_skin_type.pth"
YOLO_MODEL_PATH = "models/yolov8_skin.pt"
DB_PATH = "user_data.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- DATABASE SETUP ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    age INTEGER,
                    gender TEXT,
                    gmail TEXT,
                    skin_type TEXT,
                    skin_issues TEXT
                )''')
    conn.commit()
    conn.close()


def save_to_db(name, age, gender, gmail, skin_type, skin_issues):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO users (name, age, gender, gmail, skin_type, skin_issues) VALUES (?, ?, ?, ?, ?, ?)",
        (name, age, gender, gmail, skin_type, skin_issues)
    )
    conn.commit()
    conn.close()


# ---------------- FACE CROP USING YOLOv8 ----------------
@st.cache_resource
def load_face_detector():
    return YOLO("models/yolov8n-face.pt")


def crop_face(image_pil):
    face_model = load_face_detector()
    img = np.array(image_pil)
    results = face_model.predict(img, conf=0.25, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results and len(results[0].boxes) > 0 else []

    if len(boxes) == 0:
        st.warning("⚠️ No face detected — using full image.")
        return image_pil

    x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    pad = 60
    h, w, _ = img.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    cropped = img[y1:y2, x1:x2]
    return Image.fromarray(cropped)


# ---------------- MODEL LOADERS ----------------
@st.cache_resource
def load_vit_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3)
    model.load_state_dict(torch.load(VIT_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@st.cache_resource
def load_yolo_model():
    model = YOLO(YOLO_MODEL_PATH)
    if not hasattr(model.model, "names") or not model.model.names:
        model.model.names = {0: "pimple", 1: "wrinkle", 2: "acne", 3: "dark_spot"}
    return model


# ---------------- HELPERS ----------------
vit_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def predict_skin_type(model, image):
    img_tensor = vit_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    classes = ["dry", "oily", "combination"]
    idx = int(np.argmax(probs))
    return classes[idx], probs[idx]


# ---------------- PROBLEM DESCRIPTIONS ----------------
problem_descriptions = {
    "pimple": "🔴 **Pimples** are small inflamed spots caused by clogged pores. Try using salicylic acid or benzoyl peroxide treatments.",
    "wrinkle": "🧓 **Wrinkles** are fine lines from aging or sun exposure. Apply sunscreen daily and use retinol or peptides.",
    "acne": "⚡ **Acne** involves inflamed oil glands and can leave scars. Keep your skin clean and avoid heavy creams.",
    "dark_spot": "🌑 **Dark spots** result from hyperpigmentation. Use Vitamin C serum and sunscreen for improvement."
}


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="AI Skincare Assistant", layout="wide")
st.title("💆 AI Skincare Analyzer")
st.write("Upload or take up to 3 selfies (min 1, max 3) to detect **skin type** and **skin issues**, then get a GPT-powered skincare routine.")


init_db()

if "captured_images" not in st.session_state:
    st.session_state.captured_images = []
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# --- User Info ---
with st.expander("🧍 Enter your personal info"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    gmail = st.text_input("Gmail (Email Address)")


# --- Input method ---
input_method = st.radio("📷 Choose input method (Min 1, Max 3 Images):", ["Upload Images", "Use Camera"])

uploaded_files = []
cropped_faces = []

if input_method == "Upload Images":
    uploaded_files = st.file_uploader(
        "Upload 1 to 3 images of your face", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Please upload up to 3 images only. Truncating to the first 3.")
            uploaded_files = uploaded_files[:3]
            
        st.subheader("🖼️ Uploaded & Cropped Faces:")
        cols = st.columns(len(uploaded_files))
        for idx, uploaded_file in enumerate(uploaded_files):
            img = Image.open(uploaded_file).convert("RGB")
            cropped = crop_face(img)
            cropped_faces.append(cropped)
            with cols[idx]:
                st.image(cropped, width=150)
                st.caption(f"Image {idx + 1}")

elif input_method == "Use Camera":
    if st.session_state.captured_images:
        st.subheader("📸 Current Captures:")
        cols = st.columns(len(st.session_state.captured_images))
        for idx, cimg in enumerate(st.session_state.captured_images):
            with cols[idx]:
                st.image(cimg, width=150)
                st.caption(f"Captured Image {idx + 1}")
        cropped_faces = st.session_state.captured_images

    if len(st.session_state.captured_images) < 3:
        new_img = st.camera_input("Take a live photo")
        if new_img is not None:
            img = Image.open(new_img).convert("RGB")
            cropped = crop_face(img)
            st.session_state.captured_images.append(cropped)
            st.rerun()
    else:
        st.info("Maximum 3 images captured.")

    if st.session_state.captured_images and st.button("Clear Captures"):
        st.session_state.captured_images = []
        cropped_faces = []
        st.rerun()


st.markdown("---")

# --- Analysis & Results ---
if cropped_faces:
    if st.button("🔬 Analyze All Faces", key="analyze_button"):
        if not name or not gmail:
            st.warning("Please fill in your Name and Gmail in the personal info section before analyzing.")
        else:
            with st.spinner("Analyzing skin across images... Please wait..."):
                vit_model = load_vit_model()
                yolo_model = load_yolo_model()
                skin_types = []
                skin_confidences = []
                all_labels = set()
                annotated_imgs = []

                for face_img in cropped_faces:
                    skin_type, conf = predict_skin_type(vit_model, face_img)
                    skin_types.append(skin_type)
                    skin_confidences.append(conf)
                    

                   
                    results = yolo_model.predict(np.array(face_img), conf=0.02, verbose=False)


                    annotated = results[0].plot()
                    annotated_imgs.append(annotated)
                    cls_indices = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes is not None else []
                    labels = [results[0].names[i] for i in cls_indices]
                    all_labels.update(labels)

                most_common_skin_type = Counter(skin_types).most_common(1)[0][0]
                avg_confidence = np.mean(skin_confidences)

                st.session_state.skin_type = most_common_skin_type
                st.session_state.conf = avg_confidence
                st.session_state.labels = list(all_labels)
                st.session_state.annotated_imgs = annotated_imgs
                st.session_state.analysis_done = True
                
                save_to_db(name, age, gender, gmail, most_common_skin_type, ", ".join(all_labels))

                st.rerun()


# Display results immediately below the analysis button
if st.session_state.get("analysis_done", False):
    st.subheader("✅ Analysis Results:")
    st.success(f"🧴 Aggregated Skin Type: **{st.session_state.skin_type.capitalize()}** (Confidence: {st.session_state.conf:.2f})")

    if "annotated_imgs" in st.session_state and st.session_state.annotated_imgs:
        st.subheader("🖼️ Visual Skin Issue Detection:")
        cols = st.columns(len(st.session_state.annotated_imgs))
        for idx, ann_img in enumerate(st.session_state.annotated_imgs):
            with cols[idx]:
                st.image(ann_img, caption=f"Image {idx + 1} Issues", use_container_width=True)

    if st.session_state.labels:
        st.subheader("🔥 Key Aggregated Skin Concerns:")
        for label in st.session_state.labels:
            description = problem_descriptions.get(label, f"**{label.replace('_', ' ').capitalize()}** - Detected issue.")
            st.markdown(description)
    else:
        st.info("No common skin issues (pimple, acne, wrinkle, dark spot) were strongly detected.")
        
    st.markdown("---")

# Routine generation
if st.session_state.get("analysis_done", False):
    if st.button("🌟 Generate Personalized Routine"):
        if "skin_type" in st.session_state and st.session_state.labels is not None:
            with st.spinner("Generating product recommendations and AI routine..."):
                try:
                    df = load_ingredients()
                    recommended = recommend_ingredients(st.session_state.labels, top_n=3)
                    
                    if recommended:
                        recommended_names = [r["name"] for r in recommended]
                        st.write("---")
                        st.subheader("🧪 Recommended Ingredients/Products")
                        for r in recommended:
                            st.markdown(f"* **{r['name'].capitalize()}**: {r['function']}")
                        st.write("---")
                    else:
                        recommended_names = ["hyaluronic acid", "niacinamide", "sunscreen"]

                    routine = generate_ai_routine_gpt2(
                        st.session_state.skin_type,
                        ", ".join(st.session_state.labels),
                        recommended_names
                    )
                    st.subheader("✨ AI-Generated Routine (GPT-2)")
                    st.info(routine)
                except FileNotFoundError:
                    st.error("Error: Missing 'ingredient_recommender.py' or data.")
                except Exception as e:
                    st.error(f"An error occurred during routine generation: {e}")
        else:
            st.warning("Please analyze your skin first to get tailored recommendations and routine!")


st.markdown("---")
st.caption("Built with ❤️ using ViT + YOLOv8 + GPT-2 + Streamlit")
