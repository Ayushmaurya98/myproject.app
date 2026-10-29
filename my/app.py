import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
import os
import gdown

# 1. Page Configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier", 
    page_icon="🐱🐶", 
    layout="centered"
)

st.title("🐱🐶 Cat vs Dog VGG16 Classifier")
st.write("Upload an image to see if the model detects a Cat or a Dog.")

# 2. Google Drive Se Model Download Aur Cache Karne Ka Logic
@st.cache_resource
def load_vgg16_model():
    model_path = "model.h5"
    
    # KYA FILE PEHLE SE DOWNLOADED HAI? 
    if not os.path.exists(model_path):
        with st.spinner("Downloading 100MB+ Model from Google Drive... Please wait, this happens only once."):
            # 🔴 AAPKI GOOGLE DRIVE FILE ID YAHAN PASTE KAREIN
            FILE_ID = "1bXbgdH0WNDjOXtn_UsGuhpB5mi9bE2P7" 
            
            # Direct download URL structure banana
            url = f"https://google.com{FILE_ID}"
            
            # File download execute karna
            gdown.download(url, model_path, quiet=False)
            
    return tf.keras.models.load_model(model_path)

# Load the model
try:
    model = load_vgg16_model()
except Exception as e:
    st.error(f"Model load karne mein dikkat aayi. Kripya check karein ki File ID sahi hai aur Google Drive link 'Anyone with the link' par set hai. Error: {e}")
    st.stop()

# 3. Image Preprocessing Pipeline
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    display_img = img.copy()
    
    img = img.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    
    return display_img, processed_img

# 4. Streamlit UI Elements
uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    display_img, processed_img = preprocess_image(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Your Image")
        st.image(display_img, use_container_width=True)
        
    with col2:
        st.subheader("Prediction Result")
        with st.spinner("Running inference..."):
            predictions = model.predict(processed_img)
            
            CLASS_NAMES = ['Cat', 'Dog'] 
            
            if predictions.shape[-1] == 1:
                prob = predictions[0][0]
                if prob > 0.5:
                    result_label = "Dog"
                    confidence = prob * 100
                else:
                    result_label = "Cat"
                    confidence = (1 - prob) * 100
            else:
                predicted_class_idx = np.argmax(predictions)
                result_label = CLASS_NAMES[predicted_class_idx]
                confidence = predictions[0][predicted_class_idx] * 100
            
            st.success(f"Detected: **{result_label}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
