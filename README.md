# 🌿 Plant Disease Image Classification

## 📖 Description

Plant diseases can have a major impact on agricultural productivity and food security. This project provides an AI-powered solution for detecting plant diseases through image classification. Using a Convolutional Neural Network (CNN) model trained on leaf images, users can upload photos of plant leaves and receive instant predictions about the presence and type of disease. The model is integrated into a lightweight Flask web application for ease of use.

---

## 📁 Project Structure

Plant Disease Image Classification/
├── WebApp/
│ ├── static/
│ ├── templates/ # HTML templates
│ ├── app.py # Flask web app
│ └── plantDisease_model.h5 # Trained Keras model (not included)
├── model_training.ipynb # Jupyter notebook for training
├── requirements.txt # Dependencies
├── README.md # Project overview


---

## 🚀 Features

- Deep learning-based image classification
- Trained on plant disease datasets
- Flask web app with image upload and prediction interface
- Easy to deploy locally

---

## 🧠 Model Training

The CNN model is built and trained using Keras and TensorFlow. You can inspect and retrain the model using the provided `model_training.ipynb` notebook.

---

## 🌐 Web App

To run the web app locally:

```bash
cd WebApp
# (Optional) Activate your virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

python app.py
```

Then open your browser and go to:
http://127.0.0.1:5000
