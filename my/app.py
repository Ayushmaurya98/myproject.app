
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

st.title("🐱🐶 Cat vs Dog Image Classifier")
st.write("Upload an image to see if the model detects a Cat or a Dog.")

# 2. Download and Load Model
@st.cache_resource
def load_vgg16_model():

    model_path = "model_vgg16.h5"

    FILE_ID = "1bXbgdH0WNDjOXtn_UsGuhpB5mi9bE2P7"

    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:

        with st.spinner("Downloading model from Google Drive..."):

            # Correct Google Drive download
            downloaded_file = gdown.download(
                id=FILE_ID,
                output=model_path,
                quiet=False
            )

            if downloaded_file is None:
                raise RuntimeError(
                    "Model download failed. Check Google Drive File ID and sharing permissions."
                )

    if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
        raise RuntimeError("Model file is missing or empty.")

    # Load model
    model = tf.keras.models.load_model(model_path)

    return model


# Load model
try:
    model = load_vgg16_model()
    st.success("Model loaded successfully!")

except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# 3. Image Preprocessing
def preprocess_image(uploaded_file):

    img = Image.open(uploaded_file)

    if img.mode != "RGB":
        img = img.convert("RGB")

    display_img = img.copy()

    img = img.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    processed_img = preprocess_input(img_array)

    return display_img, processed_img


# 4. Streamlit UI
uploaded_file = st.file_uploader(
    "Choose an image file...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    display_img, processed_img = preprocess_image(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Your Image")
        st.image(display_img, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        with st.spinner("Running prediction..."):

            predictions = model.predict(processed_img, verbose=0)

            CLASS_NAMES = ["Cat", "Dog"]

            if predictions.shape[-1] == 1:

                prob = float(predictions[0][0])

                if prob > 0.5:
                    result_label = "Dog"
                    confidence = prob * 100
                else:
                    result_label = "Cat"
                    confidence = (1 - prob) * 100

            else:

                predicted_class_idx = int(np.argmax(predictions[0]))

                result_label = CLASS_NAMES[predicted_class_idx]

                confidence = float(
                    predictions[0][predicted_class_idx]
                ) 

                THRESHOLD = 0.80

                if confidence < THRESHOLD:

                    st.error(
                        "❌ Invalid or uncertain image! "
                        "Please upload only a Cat or Dog image."
                    )

                else:

                    st.success(f"Detected: {result_label} 🐱🐶")

                    st.metric(
                        label="Confidence",
                        value=f"{confidence * 100:.2f}%"
                    )

    except Exception as e:
        st.error(f"Unable to process image: {e}")
