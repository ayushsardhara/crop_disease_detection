import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import zipfile
import pandas as pd
from disease_info import disease_data

# ---------------- SETTINGS ----------------
IMG_SIZE = 224
MODEL_ZIP = "exported_model.zip"
MODEL_DIR = "exported_model"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Crop Health AI", page_icon="🌱", layout="centered")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main { background-color: #f6fff8; }

.card {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.title {
    color: #2f855a;
    font-size: 34px;
    text-align: center;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #4a5568;
    margin-bottom: 10px;
}

.badge-healthy {
    background-color: #38a169;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}

.badge-disease {
    background-color: #e53e3e;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

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

# ---------------- LOAD MODEL (KERAS 3 SAFE) ----------------
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
st.markdown("<div class='title'>🌿 Smart Crop Health Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI based Leaf Disease Detection using Deep Learning</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- UI ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown("### 🌾 Select Crop")
    selected_crop = st.selectbox("", crop_list)

    st.markdown("### 📤 Upload or Capture Image")

    input_mode = st.radio("", ["Upload Image", "Use Camera"], horizontal=True)

    uploaded_file = None
    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        uploaded_file = st.camera_input("Take a picture")

    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 🧠 Model Info")
    st.write("• CNN with Transfer Learning")
    st.write("• Dataset: PlantVillage")
    st.write("• Total Classes:", len(class_names))
    st.write("• Keras‑3 Compatible Inference")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Leaf Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("Analyzing image..."):
            out = model(img)

            if isinstance(out, dict):
                out = list(out.values())[0]

            pred = out.numpy()[0]

        crop_indices = [i for i, c in enumerate(class_names) if c.startswith(selected_crop)]
        crop_preds = [(class_names[i], float(pred[i])) for i in crop_indices]
        crop_preds.sort(key=lambda x: x[1], reverse=True)

        disease, confidence = crop_preds[0]

        is_healthy = "healthy" in disease.lower()

        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if is_healthy:
            st.markdown("<span class='badge-healthy'>✅ Healthy Leaf</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='badge-disease'>⚠️ Diseased Leaf</span>", unsafe_allow_html=True)

        st.markdown(f"### 🌿 Result: {disease}")
        st.write(f"### 🎯 Confidence: {confidence*100:.2f}%")
        st.progress(int(confidence * 100))

        # -------- BAR CHART --------
        st.markdown("### 📊 Top Predictions")

        top5 = crop_preds[:5]
        labels = [d.split("___")[1].replace("_", " ") for d, _ in top5]
        values = [v for _, v in top5]

        chart_df = pd.DataFrame({
            "Disease": labels,
            "Confidence": values
        })

        st.bar_chart(chart_df.set_index("Disease"))

        # -------- DISEASE INFO --------
        if disease in disease_data:
            st.markdown("### 📘 Disease Information")
            st.write("**Cause:**", disease_data[disease]["cause"])
            st.write("**Symptoms:**", disease_data[disease]["symptoms"])
            st.write("**Prevention:**", disease_data[disease]["prevention"])

        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.history.append((disease, round(confidence * 100, 2)))

# ---------------- HISTORY ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 🧾 Detection History (Last 5)")
for d, c in st.session_state.history[-5:][::-1]:
    st.write(f"• {d} — {c}%")
st.markdown("</div>", unsafe_allow_html=True)

st.warning("⚠️ Educational purpose only. Not a replacement for expert advice.")
st.markdown("<p style='text-align:center; font-size:13px;'>Developed by CSE Students using Deep Learning</p>", unsafe_allow_html=True)

