import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 🧠 Load trained model
model = tf.keras.models.load_model("plant_disease_model.h5")

# ✅ Define class names manually (same order used during training)
class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# 🎨 Streamlit UI
st.set_page_config(page_title="🌿 Crop Disease Detector", layout="centered")
st.title("🌿 AI Crop Disease Detector")
st.write("Upload a clear leaf photo to identify potential plant disease.")

# 📸 Upload section
uploaded_file = st.file_uploader("Upload Image or Take a Photo", type=["jpg", "jpeg", "png"])

# 🖼️ Show and process image
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 🔍 Predict
    predictions = model.predict(img_array)[0]
    top_3_indices = predictions.argsort()[-3:][::-1]
    top_3_classes = [class_names[i] for i in top_3_indices]
    top_3_confidence = [predictions[i] * 100 for i in top_3_indices]

    # 🎯 Show results
    st.subheader("🔍 Top 3 Predictions")
    for i in range(3):
        st.write(f"**{top_3_classes[i]}** — {top_3_confidence[i]:.2f}%")
else:
    st.info("📷 Upload a photo to begin diagnosis.")
