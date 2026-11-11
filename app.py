
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Smart Waste Segregation AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ Smart Waste Segregation AI")
st.write("Upload a waste image to identify its category.")

MODEL_PATH = "model/waste_model.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run train.py first.")
    st.stop()

model = load_model()

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Waste Image",
             use_container_width=True)

    if st.button("Detect Waste"):
        resized = image.resize((224, 224))

        image_array = np.array(resized)
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)

        class_names = [
            "cardboard", "glass", "metal",
            "organic", "paper", "plastic"
        ]

        predicted_index = np.argmax(predictions[0])
        category = class_names[predicted_index]
        confidence = float(predictions[0][predicted_index]) * 100

        st.success(f"Detected Category: {category.upper()}")
        st.metric("Confidence", f"{confidence:.2f}%")

        if category == "organic":
            st.info("Suggestion: Put in an organic waste bin.")
        else:
            st.info(f"Suggestion: Put in the {category} recycling bin.")

