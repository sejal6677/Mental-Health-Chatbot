import json
import numpy as np
import random
import nltk
import pickle
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from nltk.stem import WordNetLemmatizer
import re
import os

# Download necessary NLTK resources if not already downloaded
nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

lemmatizer = WordNetLemmatizer()

import os

# Load the intents file with correct path
intents_path = os.path.join(os.path.dirname(__file__), "backend", "intents.json")
with open(intents_path) as file:
    intents = json.load(file)

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Tokenize and prepare training data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        word_list = re.findall(r'\b\w+\b', pattern.lower())
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and sort vocabulary
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes for inference
words_path = os.path.join(os.path.dirname(__file__), "words.pkl")
classes_path = os.path.join(os.path.dirname(__file__), "classes.pkl")
pickle.dump(words, open(words_path, "wb"))
pickle.dump(classes, open(classes_path, "wb"))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]
    for w in words:
        bag.append(1 if w in pattern_words else 0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

# Build neural network
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile and train the model
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

if __name__ == "__main__":
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
    # Save the trained model
    model_path = os.path.join(os.path.dirname(__file__), "chatbot_model.h5")
    model.save(model_path)
    print("Model trained and saved as chatbot_model.h5")
