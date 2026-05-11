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
    "Simple RNN": keras.models.load_model("final_RNN_model.keras"),
    "LSTM": keras.models.load_model("final_LSTM_model.keras"),
    "GRU": keras.models.load_model("final_GRU_model.keras")
}

# =========================================
# Parameters (MUST match training)
# =========================================
max_len = 300

# =========================================
# Preprocessing (IDENTICAL to training)
# =========================================
#Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    # text = re.sub(r"\s+", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    text = re.sub(r"\d", '', text)
    text = text.lower()
    return text

# =========================================
# Prediction Function
# =========================================
def predict_sentiment(review, model):

    # preprocess
    cleaned = denoise_text(review)

    # tokenize
    seq = tokenizer.texts_to_sequences([cleaned])

    # ⚠️ IMPORTANT: same padding as training (pre)
    padded = keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    
    # predict
    pred = model.predict(padded, verbose=0)

    score = float(pred[0][0])

    sentiment = "Positive" if score > 0.5 else "Negative"
    confidence = score if score > 0.5 else (1 - score)

    return sentiment, confidence, cleaned, seq, pred

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

            sentiment, confidence, cleaned, seq, pred = predict_sentiment(
                review,
                selected_model
            )

        st.subheader("Result")

        st.success(f"Model: {selected_model_name}")
        st.success(f"Sentiment: {sentiment}")
        st.info(f"Confidence: {confidence:.4f}")

        # =========================================
        # Debug Section (VERY IMPORTANT)
        # =========================================
        with st.expander("🔍 Debug Info"):
            st.write("Cleaned Text:", cleaned)
            st.write("Sequence:", seq)
            st.write("Raw Prediction:", pred)
