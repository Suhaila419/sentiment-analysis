import streamlit as st
import numpy as np
import re
import pickle

from bs4 import BeautifulSoup
from tensorflow import keras


# =========================================
# Load Tokenizer
# =========================================

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# =========================================
# Load Models
# =========================================

models = {
    "Simple RNN": keras.models.load_model("best_RNN_model.keras"),
    "LSTM": keras.models.load_model("best_LSTM_model.keras"),
    "GRU": keras.models.load_model("best_GRU_model.keras")
}

# =========================================
# Parameters (must match training)
# =========================================

max_len = 300

# =========================================
# Preprocessing (IDENTICAL to training)
# =========================================

def denoise_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'\[[^]]*\]', '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d", '', text)
    text = text.lower().strip()
    return text

# =========================================
# Prediction Function
# =========================================

def predict_sentiment(review, model):

    # preprocess
    review = denoise_text(review)

    # tokenize
    seq = tokenizer.texts_to_sequences([review])
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    # predict
    pred = model.predict(padded, verbose=0)
    print("RAW PRED:", pred)

    # =====================================
    # Sigmoid output (1 neuron)
    # =====================================
    score = float(pred[0][0])
    sentiment = "Positive" if score > 0.5 else "Negative"
    confidence = score if score > 0.5 else 1 - score

    return sentiment, confidence

# =========================================
# Streamlit UI
# =========================================

st.set_page_config(page_title="IMDB Sentiment Analysis", layout="centered")

st.title("🎬 IMDB Sentiment Analysis")
st.write("Choose a model and enter a movie review.")

# =========================================
# Model Selection
# =========================================

selected_model_name = st.selectbox(
    "Select Model",
    list(models.keys())
)

selected_model = models[selected_model_name]

# =========================================
# Input
# =========================================

review = st.text_area("Enter Movie Review", height=250)

# =========================================
# Predict
# =========================================

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        with st.spinner("Analyzing review..."):

            sentiment, confidence = predict_sentiment(
                review,
                selected_model
            )

        st.subheader("Result")

        st.success(f"Model: {selected_model_name}")
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence:.4f}")