import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Load the trained model
model = load_model("fruit_freshness_model.keras", compile=False)

# Class labels (must match your model)
class_names = ['freshapples', 'freshbanana', 'freshoranges', 
               'rottenapples', 'rottenbanana', 'rottenoranges']

st.set_page_config(page_title="Fruit Freshness Detector", layout="centered")
st.title("🍎🥭 Fruit Freshness Detector")
st.write("Upload an image of a fruit and we'll tell you if it's **Fresh** or **Rotten**.")

# File uploader
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"**Prediction:** {class_names[predicted_class]}")
    st.info(f"**Confidence:** {confidence*100:.2f}%")
