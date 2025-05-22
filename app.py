import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the saved model
model = load_model("cnn_best_model.h5")

# Class names (ensure these match your training data)
class_names = ['cloudy', 'rain', 'shine', 'sunrise']  # Update if needed

# Set title
st.title("Weather Image Classifier")
st.write("Upload a weather image (cloudy, rain, shine, sunrise) and see the prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image_display.resize((150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    # Display result
    st.markdown(f"### Predicted Class: **{predicted_class.upper()}**")
