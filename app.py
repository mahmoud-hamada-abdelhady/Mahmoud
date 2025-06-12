import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from huggingface_hub import hf_hub_download

# Load trained model
model_path = hf_hub_download(repo_id="mahmoudhamada11/brain-tumor-vgg16", filename="vgg16_final_model.keras")

model = load_model(model_path)

# Class names (order should match your training data)
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Streamlit App Title
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI image and the AI model will classify the type of tumor (if any).")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show result
    st.success(f"Predicted class: **{predicted_class}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")
