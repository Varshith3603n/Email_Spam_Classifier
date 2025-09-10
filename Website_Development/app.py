import os
import streamlit as st
import pickle
import string
import nltk

# Download NLTK data (only if not already present)
nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Initialize stemmer
ps = PorterStemmer()

# ✅ Get absolute path of current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load vectorizer and model safely
with open(os.path.join(BASE_DIR, 'vectorizer.pkl'), 'rb') as f:
    tfidf = pickle.load(f)

with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
    model = pickle.load(f)


# --------------------------
# Text preprocessing function
# --------------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in string.punctuation and i not in stopwords.words('english'):
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


# --------------------------
# Streamlit UI
# --------------------------
st.title("📧 Email Spam Classifier")

input_sms = st.text_input("✍️ Enter your message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms]).toarray()

        # Predict
        result = model.predict(vector_input)[0]

        # Show result
        if result == 1:
            st.error("🚨 This message is **Spam**")
        else:
            st.success("✅ This message is **Ham (Not Spam)**")
