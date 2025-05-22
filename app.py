import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

st.title("🌦️ Weather Image Classifier")

# Debug
st.write("Files in directory:", os.listdir())

# Load model
try:
    model = load_model("cnn_best_model.h5")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

class_names = ['cloudy', 'rain', 'shine', 'sunrise']

uploaded_file = st.file_uploader("Upload a weather image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        st.markdown(f"### 🧠 Predicted Class: **{predicted_class.upper()}**")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
