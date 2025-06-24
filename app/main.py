import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from utils import load_model, predict_class

st.set_page_config(page_title="Star Classifier 🌠", layout="centered")

# Load model
model = load_model('../model/star_galaxy_classifier_using_cnn.pt')
model.eval()

# UI
st.title("Galaxy or a star")
st.write("Upload an image of an astromical body and check whether it is a star or a galaxy.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Classify"):
        label, confidence = predict_class(image, model)
        st.success(f"Prediction: **{label}**")
        st.write(f"Confidence: {confidence:.2f}")
