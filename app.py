import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import zipfile
from disease_info import disease_data

# ---------------- SETTINGS ----------------
IMG_SIZE = 224
MODEL_ZIP = "exported_model.zip"
MODEL_DIR = "exported_model"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Crop Health AI", page_icon="🌱", layout="centered")

# ---------------- CLASS NAMES ----------------
class_names = [
'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
'Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
'Strawberry___Leaf_scorch','Strawberry___healthy',
'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold',
'Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite',
'Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus',
'Tomato___healthy'
]

# ---------------- LOAD MODEL USING TFSMLAYER ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(MODEL_ZIP, "r") as z:
            z.extractall()

    layer = tf.keras.layers.TFSMLayer(
        MODEL_DIR, call_endpoint="serving_default"
    )

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        layer
    ])
    return model

model = load_model()

# ---------------- CROP LIST ----------------
crop_list = sorted(list(set([c.split("___")[0] for c in class_names])))

# ---------------- HEADER ----------------
st.markdown("<h1 style='text-align:center;'>🌿 Smart Crop Health Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI based leaf disease detection using Deep Learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UI ----------------
col1, col2 = st.columns(2)

with col1:
    selected_crop = st.selectbox("🌾 Select Crop", crop_list)
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])

with col2:
    st.markdown("### 🧠 Model Info")
    st.write("CNN Model: MobileNetV2")
    st.write("SavedModel via TFSMLayer (Keras 3 compatible)")
    st.write("Total Classes:", len(class_names))

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("Analyzing image..."):
            out = model(img)

# If model returns dict (SavedModel), get first value
if isinstance(out, dict):
    out = list(out.values())[0]

pred = out.numpy()[0]


        crop_indices = [i for i, c in enumerate(class_names) if c.startswith(selected_crop)]
        crop_preds = [(class_names[i], pred[i]) for i in crop_indices]
        crop_preds.sort(key=lambda x: x[1], reverse=True)

        disease, confidence = crop_preds[0]

        st.success(f"🌿 Disease Detected: {disease}")
        st.progress(int(confidence * 100))
        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.session_state.history.append((disease, round(confidence * 100, 2)))

        if disease in disease_data:
            st.markdown("### 📘 Disease Information")
            st.write("**Cause:**", disease_data[disease]["cause"])
            st.write("**Symptoms:**", disease_data[disease]["symptoms"])
            st.write("**Prevention:**", disease_data[disease]["prevention"])
        else:
            st.info("No additional disease information available.")

# ---------------- HISTORY ----------------
st.markdown("---")
st.markdown("### 🧾 Detection History (Last 5)")
for d, c in st.session_state.history[-5:][::-1]:
    st.write(f"• {d} — {c}%")

st.warning("⚠️ Educational purpose only. Not a replacement for expert advice.")
st.markdown("<p style='text-align:center; font-size:13px;'>Developed by CSE Students using Deep Learning</p>", unsafe_allow_html=True)

