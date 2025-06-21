import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download necessary resources only if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

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
        if i not in stop_words:
            y.append(ps.stem(i))

    return " ".join(y)

# ✅ Load vectorizer and model (ensure they were trained together)
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('mmodel.pkl', 'rb'))  # Use correct name

# 🖼️ Streamlit App UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a message to check.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        st.write("🔍 Transformed Text:", transformed_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]
        st.write("🔢 Raw Prediction (0 = Not Spam, 1 = Spam):", result)

        # 4. Display result
        if result == 1:
            st.header("📛 Spam")
        else:
            st.header("✅ Not Spam")



