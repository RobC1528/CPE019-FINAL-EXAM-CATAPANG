import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from PIL import Image
import os

# Debug file check
st.write("Current files:", os.listdir())

# Load the saved model
try:
    model = load_model("cnn_best_model.h5")
except Exception as e:
    st.error(f" Failed to load model: {e}")
    st.stop()

# Class names (must match training labels)
class_names = ['cloudy', 'rain', 'shine', 'sunrise']

# App title and description
st.title("Weather Image Classifier")
st.write("Upload a weather image (cloudy, rain, shine, sunrise) and see the prediction.")

# Upload image
uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image_display.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        st.markdown(f"### Predicted Class: **{predicted_class.upper()}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
