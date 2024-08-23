import streamlit as st
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import joblib  # for loading the trained model

# Load the trained SVM model and label encoder
model = joblib.load('svm_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Set the image size expected by the model
image_size = (64, 64)

def classify_image(image):
    # Preprocess the image
    img = cv2.resize(image, image_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img.flatten().reshape(1, -1)  # Flatten and reshape
    
    # Predict using the SVM model
    prediction = model.predict(img)
    label = label_encoder.inverse_transform(prediction)[0]
    
    return label

# Streamlit UI
st.title("Cat vs. Dog Classifier")
st.write("Upload an image to classify whether it's a cat or a dog.")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Predict the class
    label = classify_image(image)
    st.write(f"Prediction: **{label.capitalize()}**")
