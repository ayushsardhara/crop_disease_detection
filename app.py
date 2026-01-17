import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from disease_info import disease_data

IMG_SIZE = 224
MODEL_PATH = "crop_disease_model.h5"
DATASET_PATH = "plantvillage_dataset/color"

st.set_page_config(page_title="Smart Crop Health AI", page_icon="🌱")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = sorted(os.listdir(DATASET_PATH))

# Crop list
crop_list = sorted(list(set([c.split("___")[0] for c in class_names])))

st.title("🌿 Smart Crop Health Detection System")

# ---- HISTORY ----
if "history" not in st.session_state:
    st.session_state.history = []

# ---- UI ----
col1, col2 = st.columns(2)

with col1:
    selected_crop = st.selectbox("🌾 Select Crop", crop_list)
    uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg","png","jpeg"])

with col2:
    st.markdown("### 🧠 Model Info")
    st.write("CNN: MobileNetV2")
    st.write("Accuracy: ~95%")
    st.write("Classes:", len(class_names))

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    if st.button("🔍 Detect Disease"):
        pred = model.predict(img)[0]

        # Filter by crop
        crop_indices = [i for i,c in enumerate(class_names) if c.startswith(selected_crop)]
        crop_preds = [(class_names[i], pred[i]) for i in crop_indices]
        crop_preds.sort(key=lambda x: x[1], reverse=True)

        disease, confidence = crop_preds[0]

        st.success(f"🌿 Disease: {disease}")
        st.progress(int(confidence*100))
        st.write(f"Confidence: {confidence*100:.2f}%")

        # Save history
        st.session_state.history.append((disease, round(confidence*100,2)))

        # Disease info
        if disease in disease_data:
            st.markdown("### 📘 Disease Information")
            st.write("**Cause:**", disease_data[disease]["cause"])
            st.write("**Symptoms:**", disease_data[disease]["symptoms"])
            st.write("**Prevention:**", disease_data[disease]["prevention"])
        else:
            st.info("No additional info available for this disease.")

# ---- HISTORY PANEL ----
st.markdown("---")
st.markdown("### 🧾 Detection History")

for d,c in st.session_state.history[-5:][::-1]:
    st.write(f"• {d} — {c}%")

st.warning("⚠️ Educational purpose only. Not a replacement for expert advice.")
