# Chatbot Project

## Overview
This project is a chatbot application that uses natural language processing (NLP) and machine learning techniques to understand and respond to user inputs. The chatbot is trained using intents defined in a JSON file and a neural network model built with TensorFlow Keras.

## Features
- Intent classification using a neural network
- Text preprocessing with NLTK (tokenization, lemmatization)
- Model training and saving for inference
- Easily extendable intents via JSON configuration

## Project Structure
- `train_model.py`: Script to preprocess data, train the neural network model, and save the trained model.
- `backend/intents.json`: JSON file containing chatbot intents, patterns, and responses.
- `test_tf_import.py`: Test script to verify TensorFlow Keras imports.
- `requirements.txt`: Python dependencies required for the project.

## Installation
1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Download necessary NLTK data (handled automatically in `train_model.py`).

## Usage
- To train the chatbot model, run:
  
  python train_model.py

- The trained model will be saved as `chatbot_model.h5`.
To run the trained model,
  python backend/app.py

## Troubleshooting
- If you encounter issues with TensorFlow imports, ensure TensorFlow is installed correctly and your environment is activated.

Chatbot image:
![Image](https://github.com/user-attachments/assets/db013cbc-8366-4a06-a810-5383ab463dc7)
- Use `test_tf_import.py` to verify TensorFlow Keras imports:
  ```
  python test_tf_import.py
  ```


