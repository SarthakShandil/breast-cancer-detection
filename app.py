import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('breast_cancer_model.h5')

# Map prediction index to class name
label_map = {0: 'Benign', 1: 'Malignant', 2: 'Normal'}

# Title
st.title("🧠 Breast Cancer Detection from Ultrasound")

# File uploader
uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to grayscale
    image_pil = Image.open(uploaded_file).convert("L")
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Convert to numpy array and preprocess
    img = np.array(image_pil)
    img = cv2.resize(img, (224, 224))  # Resize to model input size
    img = img / 255.0                  # Normalize
    img = img.reshape(1, 224, 224, 1)  # Add batch & channel dims

    # Predict when button is clicked
    if st.button("Predict"):
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.markdown(f"### 🩺 Prediction: **{label_map[predicted_class]}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")
