
# 4. Streamlit UI
uploaded_file = st.file_uploader(
    "Upload a Cat or Dog image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        display_img, processed_img = preprocess_image(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your Image")
            st.image(display_img, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")

            with st.spinner("Checking image..."):
                predictions = model.predict(processed_img, verbose=0)

                CLASS_NAMES = ["Cat", "Dog"]

                if predictions.shape[-1] == 1:
                    prob = float(predictions[0][0])

                    if prob > 0.5:
                        result_label = "Dog"
                        confidence = prob
                    else:
                        result_label = "Cat"
                        confidence = 1 - prob

                else:
                    predicted_class_idx = int(np.argmax(predictions[0]))
                    result_label = CLASS_NAMES[predicted_class_idx]
                    confidence = float(predictions[0][predicted_class_idx])

                THRESHOLD = 0.80

                if confidence < THRESHOLD:
                    st.error(
                        "Invalid or uncertain image! "
                        "Please upload only a Cat or Dog image."
                    )
                else:
                    st.success(f"Detected: {result_label}")

                    st.metric(
                        label="Confidence",
                        value=f"{confidence * 100:.2f}%"
                    )

    except Exception as e:
        st.error(f"Unable to process image: {e}")
