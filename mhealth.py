import numpy as np

import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Conv1D, GlobalMaxPooling1D, Flatten, Input
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import random
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Setting random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Define global variables
VOCAB_SIZE = 10000
MAX_SENTENCE_LENGTH = 10

# Function to read and preprocess data
def load_and_preprocess_data(filepath):
    dataset = pd.read_csv(filepath, low_memory=False)
    dataset = dataset[['text', 'trans_text', 'id', 'name']]
    mh_ids = [6, 39, 50, 64, 65, 66, 67, 68, 69, 74]
    dataset.loc[dataset['id'].isin(mh_ids), 'name'] = 1  # Yes mental health
    dataset.loc[~dataset['id'].isin(mh_ids), 'name'] = 0  # No mental health
    dataset['name'] = dataset['name'].astype('category')
    return dataset

# Function to clean and tokenize text
def clean_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'[^\w\s]', '', text)
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Function to vectorize text
def vectorize_text(data, max_tokens=VOCAB_SIZE, output_sequence_length=MAX_SENTENCE_LENGTH):
    text_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=output_sequence_length)
    text_vectorizer.adapt(data)
    return text_vectorizer

# Function to build and compile model
def build_model(model_type, vectorizer, vocab_size=VOCAB_SIZE):
    inputs = Input(shape=(1,), dtype=tf.string)
    x = vectorizer(inputs)
    x = Embedding(input_dim=vocab_size, output_dim=128)(x)
    
    if model_type == 'FNN':
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        x = Flatten()(x)
    elif model_type == 'LSTM':
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
    elif model_type == 'Conv1D':
        x = Conv1D(64, 8, activation="relu", padding="same")(x)
        x = GlobalMaxPooling1D()(x)
    
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# Function to plot results
def plot_results(models, results):
    fig, ax = plt.subplots(figsize=(8, 5))
    results_df = pd.DataFrame(results).T
    results_df.plot(kind="barh", ax=ax)
    ax.set_title('Model Performance')
    ax.set_xlabel('Score')
    ax.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

# Function to evaluate model
def evaluate_model(model, dataset):
    pred_probs = model.predict(dataset)
    preds = tf.squeeze(tf.round(pred_probs))
    accuracy = accuracy_score(y_true, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1}

# Main function to execute the pipeline
def main():
    # Load and preprocess the data
    filepath = "/path/to/your/dataset.csv"
    data = load_and_preprocess_data(filepath)
    data['clean_text'] = data['trans_text'].apply(clean_text)
    
    # Split data
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_sentences = train_data['clean_text'].tolist()
    train_labels = train_data['name'].tolist()
    
    # Prepare text vectorizer
    vectorizer = vectorize_text(train_sentences)
    
    # Build and compile models
    fnn_model = build_model('FNN', vectorizer)
    lstm_model = build_model('LSTM', vectorizer)
    conv1d_model = build_model('Conv1D', vectorizer)
    
    # Train models
    fnn_model.fit(train_data, epochs=5, validation_split=0.1)
    lstm_model.fit(train_data, epochs=5, validation_split=0.1)
    conv1d_model.fit(train_data, epochs=5, validation_split=0.1)
    
    # Evaluate models
    fnn_results = evaluate_model(fnn_model, test_data)
    lstm_results = evaluate_model(lstm_model, test_data)
    conv1d_results = evaluate_model(conv1d_model, test_data)
    
    # Plot results
    plot_results(['FNN', 'LSTM', 'Conv1D'], [fnn_results, lstm_results, conv1d_results])

if __name__ == "__main__":
    main()
