import streamlit as st
import numpy as np
import pickle
import re
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Google Drive direct link
MODEL_URL = "https://drive.google.com/uc?id=1-EMkIIZN_A5Fif7IghXvYtcXe2uTQlVX"


def download_model():
    if not os.path.exists("final_model.h5"):
        r = requests.get(MODEL_URL)
        with open("final_model.h5", "wb") as f:
            f.write(r.content)

download_model()

# Load model
model = load_model("final_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

MAX_SEQUENCE_LENGTH = 100

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Prediction
def predict_text(text):
    cleaned = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)

    pred = model.predict(padded, verbose=0)[0]

    confidence = np.max(pred)
    label = label_encoder.classes_[np.argmax(pred)]

    return label, confidence

# UI
st.title("🧠 Mental Health Prediction")

text = st.text_area("Enter text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        result, confidence = predict_text(text)
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}")
