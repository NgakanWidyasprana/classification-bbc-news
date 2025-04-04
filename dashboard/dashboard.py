
# Import Library
import re
import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def data_cleaning(sentence):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    text = sentence.lower()
    text = re.sub(r'http\S+|www\S+', '', sentence)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = sentence.split()
    no_stopwords = [word for word in words if word not in stopwords]
    cleaned_text = " ".join(no_stopwords)

    return cleaned_text

def data_lemmatization(lemmatizer, sentence):
    lemmatized_words = [lemmatizer.lemmatize(word, pos='v') for word in sentence]
    lemmatized_sentence = ' '.join(lemmatized_words)

    return lemmatized_sentence

def preprocess_input(title, tokenizer, maxlen, padding):
    title_seq = tokenizer.texts_to_sequences([title])
    title_padded = pad_sequences(title_seq, maxlen=maxlen, padding=padding)
    return title_padded

def get_label(predictions, tokenizer):
    predicted_indices = np.argmax(predictions, axis=1)
    index_to_label = {v - 1: k for k, v in label_tokenizer.word_index.items()}
    predicted_labels = [index_to_label[idx] for idx in predicted_indices]
    return predicted_labels

# Load the model from .keras format and variabel
model = tf.keras.models.load_model('news_classification_model.keras')
PADDING = 'post'
MAXLEN = 150

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_tokenizer.pkl', 'rb') as f:
    label_tokenizer = pickle.load(f)

# Set up the lemmatizer
lemmatizer = WordNetLemmatizer()

# Main Process
# Streamlit app layout
st.title("Title Category Prediction")
st.write("Enter a news article title, and we'll predict what category it's about.")

# User input
title_input = st.text_input("Enter title:")

# Preprocessing text-input
if title_input:
    # Main process - preprocessing input
    cleaned_text = data_cleaning(title_input)
    tokenized_text = word_tokenize(cleaned_text)
    lemmatized_text = data_lemmatization(lemmatizer, tokenized_text)

    # Main process - make predictions and get labels
    preprocessed_input = preprocess_input(lemmatized_text, tokenizer, MAXLEN, PADDING)
    predictions=model.predict(preprocessed_input)
    labels = get_label(predictions, label_tokenizer)

    # Main process - show the output
    st.write("Predicted Category:", labels[0])
