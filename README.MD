# Star–Galaxy Classifier

This is a **deep learning-powered image classification app** that distinguishes between **stars** and **galaxies** using a simple **Convolutional Neural Network (CNN)** trained on telescope imagery.

The project is designed to be a minimal, complete demo for showcasing a trained computer vision model in action, using **Streamlit** as the frontend and **PyTorch** for model inference.

This project was built as part of my hands-on practice in **deep learning** and model deployment.

---

## Dataset
The model was trained on real world dataset. 
Dataset link: https://www.kaggle.com/datasets/divyansh22/dummy-astronomy-data 

---

## 🔭 What it Does

- Accepts a telescope image (JPEG/PNG) from the user via a web interface
- Uses a CNN trained from scratch to classify the image as either:
  - 🌟 **Star**
  - 🌌 **Galaxy**
- Displays the predicted class and model confidence

> 📝 This model is trained on a dataset of telescope images and serves as a demo for deploying simple computer vision models.

---

## 🚀 Tech Stack

- **PyTorch** — for CNN model and inference
- **Streamlit** — for interactive web UI
- **Pillow / torchvision** — for image preprocessing and transforms
