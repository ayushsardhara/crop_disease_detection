import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from disease_info import disease_data

# ---------------- SETTINGS ----------------
IMG_SIZE = 224
MODEL_ZIP = "exported_model.zip"
MODEL_DIR = "exported_model"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Crop Health AI", page_icon="🌱", layout="centered")

# ---------------- CSS ----------------
st.markdown("""
<style>
.main { background-color: #f6fff8; }

.card {
    background-color: white;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

.title {
    color: #2f855a;
    font-size: 30px;
    text-align: center;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #4a5568;
    margin-bottom: 8px;
}

.badge-healthy {
    background-color: #38a169;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
}

.badge-disease {
    background-color: #e53e3e;
    color: white;
    padding: 6px 14px;
    border-radius: 20px;
    font-weight: bold;
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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_DIR):
        with zipfile.ZipFile(MODEL_ZIP, "r") as z:
            z.extractall()

    layer = tf.keras.layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(224, 224, 3)),
        layer
    ])
    return model

model = load_model()

# ---------------- GRAD-CAM ----------------
def make_gradcam_heatmap(img_array, model):
    img_tensor = tf.convert_to_tensor(img_array)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        if isinstance(preds, dict):
            preds = list(preds.values())[0]
        class_channel = preds[:, tf.argmax(preds[0])]

    grads = tape.gradient(class_channel, img_tensor)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, img_tensor[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap * 0.4 + img
    return np.uint8(superimposed)

# ---------------- CROP LIST ----------------
crop_list = sorted(list(set([c.split("___")[0] for c in class_names])))

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🌿 Smart Crop Health Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI based Leaf Disease Detection using Deep Learning</div>", unsafe_allow_html=True)

# ---------------- UI (MOBILE FRIENDLY) ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
selected_crop = st.selectbox("🌾 Select Crop", crop_list)

mode = st.radio("Image Input", ["Upload Image", "Use Camera"], horizontal=True)

uploaded_file = None
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "png", "jpeg"])
else:
    uploaded_file = st.camera_input("Capture Leaf Image")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Input Leaf", use_container_width=True)

    img = np.array(image)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized / 255.0
    img_input = np.expand_dims(img_norm, axis=0)

    if st.button("🔍 Detect Disease", use_container_width=True):
        with st.spinner("Analyzing image..."):
            out = model(img_input)
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
        top5 = crop_preds[:5]
        labels = [d.split("___")[1].replace("_", " ") for d, _ in top5]
        values = [v for _, v in top5]

        chart_df = pd.DataFrame({"Disease": labels, "Confidence": values})
        st.bar_chart(chart_df.set_index("Disease"))

        # -------- GRAD-CAM --------
        st.markdown("### 🔥 Model Focus (Heatmap)")
        heatmap = make_gradcam_heatmap(img_input, model)
        overlay = overlay_heatmap(img, heatmap)
        st.image(overlay, caption="Grad‑CAM Heatmap", use_container_width=True)

        # -------- INFO --------
        if disease in disease_data:
            st.markdown("### 📘 Disease Information")
            st.write("**Cause:**", disease_data[disease]["cause"])
            st.write("**Symptoms:**", disease_data[disease]["symptoms"])
            st.write("**Prevention:**", disease_data[disease]["prevention"])

        st.markdown("</div>", unsafe_allow_html=True)

st.warning("⚠️ Educational purpose only. Not a replacement for expert advice.")
