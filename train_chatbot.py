import json
import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents.json") as file:
    intents = json.load(file)

# Prepare training data
sentences = []
labels = []
classes = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        sentences.append(pattern)
        labels.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

# Preprocess sentences
def preprocess(text):
    words = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words])

sentences = [preprocess(s) for s in sentences]

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)
y = np.array(labels)

# Train the Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

# Save model and data
pickle.dump(model, open("nb_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))
