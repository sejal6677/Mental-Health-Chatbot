# chatbot_nn.py

import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import re
import os
import sys

# Initialize
lemmatizer = WordNetLemmatizer()

import os

# Load data and model
intents_path = os.path.join(os.path.dirname(__file__), "backend", "intents.json")
words_path = os.path.join(os.path.dirname(__file__), "words.pkl")
classes_path = os.path.join(os.path.dirname(__file__), "classes.pkl")
model_path = os.path.join(os.path.dirname(__file__), "chatbot_model.h5")
nb_model_path = os.path.join(os.path.dirname(__file__), "nb_model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

intents = json.load(open(intents_path))
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))

# Check if TensorFlow/Keras is available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    
    # Try to load the model if it exists
    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Neural network model loaded successfully.")
    else:
        print("Model file 'chatbot_model.h5' not found.")
        print("Using fallback to the existing model 'nb_model.pkl'")
        model = pickle.load(open(nb_model_path, "rb"))
        vectorizer = pickle.load(open(vectorizer_path, "rb"))
except ImportError:
    print("TensorFlow/Keras not available. Using fallback model.")
    model = pickle.load(open(nb_model_path, "rb"))
    vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Preprocess user input
def preprocess(sentence):
    sentence_words = re.findall(r'\b\w+\b', sentence.lower())
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Create a bag-of-words
def bag_of_words(sentence, words):
    sentence_words = preprocess(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

# Predict class
def predict_class(sentence):
    # Check if we're using the neural network model or the fallback model
    if 'tensorflow' in sys.modules and hasattr(model, 'predict'):
        # Neural network model
        bow = bag_of_words(sentence, words)
        res = model.predict(np.array([bow]))[0]
        threshold = 0.25
        results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(res) if prob > threshold]
        results.sort(key=lambda x: float(x["probability"]), reverse=True)
        return results
    else:
        # Fallback model (nb_model)
        sentence_processed = preprocess(sentence)
        sentence_text = " ".join(sentence_processed)
        X = vectorizer.transform([sentence_text])
        probs = model.predict_proba(X)[0]
        results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(probs)]
        results.sort(key=lambda x: float(x["probability"]), reverse=True)
        return results[:1]  # Only return top result

# Get response
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I do not understand..."
    tag = intents_list[0]['intent']
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I do not understand..."

# Chatbot wrapper
def chatbot_response(msg):
    intents_list = predict_class(msg)
    print("Predicted intent:", intents_list)  # Debug line
    return get_response(intents_list, intents)

# Command-line loop
def main():
    print("Chatbot (NN) is running! (type 'quit' to stop)")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(msg))

if __name__ == "__main__":
    main()
