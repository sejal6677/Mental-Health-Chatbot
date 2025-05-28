import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

# Load saved model and data
model = pickle.load(open("nb_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

with open("intents.json") as file:
    intents = json.load(file)

# Preprocess user input
def preprocess(text):
    words = word_tokenize(text.lower())
    return " ".join([lemmatizer.lemmatize(w) for w in words])

# Predict class
def predict_class(sentence):
    sentence = preprocess(sentence)
    X = vectorizer.transform([sentence])
    probs = model.predict_proba(X)[0]
    results = [{"intent": classes[i], "probability": str(prob)} for i, prob in enumerate(probs)]
    results.sort(key=lambda x: float(x["probability"]), reverse=True)
    return results[:1]     # Only return top result

# Get a response based on the intent
def get_response(intents_list, intents_json):
    if not intents_list:
        return "I do not understand..."
    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I do not understand..."

# Wrapper function
def chatbot_response(text):
    intents_list = predict_class(text)
    print("Predicted intent:", intents_list)
    return get_response(intents_list, intents)

# CLI main loop
def main():
    print("Chatbot is running! (type 'quit' to stop)")
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            print("Chatbot: Goodbye!")
            break
        print("Chatbot:", chatbot_response(msg))

if __name__ == "__main__":
    main()
