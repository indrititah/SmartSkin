import json
import random
import nltk
import string
import numpy as np
import pickle
import os
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

global responses, lemmatizer, tokenizer, le, model, input_shape
input_shape = 13

# Cetak pesan log saat mengunduh sumber daya nltk
print("Mengunduh sumber daya untuk 'punkt'...")
nltk.download('punkt', quiet=True)
print("Unduhan sumber daya untuk 'punkt' selesai.")

print("Mengunduh sumber daya untuk 'wordnet'...")
nltk.download('wordnet', quiet=True)
print("Unduhan sumber daya untuk 'wordnet' selesai.")

print("Mengunduh sumber daya untuk 'omw-1.4'...")
nltk.download('omw-1.4', quiet=True)
print("Unduhan sumber daya untuk 'omw-1.4' selesai.")

# import dataset answer
def load_response():
    global responses
    responses = {}
    with open('model2/dataset.json') as content:
        data = json.load(content)
    for intent in data['intents']:
        responses[intent['tag']]=intent['responses']

# import model
def preparation():
    load_response()
    global lemmatizer, tokenizer, le, model
    tokenizer = pickle.load(open('model2/tokenizer.pkl', 'rb'))
    le = pickle.load(open('model2/labelencoder.pkl', 'rb'))
    model = keras.models.load_model('model2/chat_model.h5')
    lemmatizer = WordNetLemmatizer()
    print("Tokenizer loaded successfully.")

# hapus tanda baca
def remove_punctuation(text):
    texts_p = []
    text = [letters.lower() for letters in text if letters not in string.punctuation]
    text = ''.join(text)
    texts_p.append(text)
    return texts_p

# mengubah text menjadi vector
def vectorization(texts_p):
    vector = tokenizer.texts_to_sequences(texts_p)
    vector = np.array(vector).reshape(-1)
    vector = pad_sequences([vector], input_shape)
    return vector

# klasifikasi pertanyaan user
def predict(vector):
    output = model.predict(vector)
    output = output.argmax()
    response_tag = le.inverse_transform([output])[0]
    return response_tag

# menghasilkan jawaban berdasarkan pertanyaan user
def generate_response(text):
    texts_p = remove_punctuation(text)
    vector = vectorization(texts_p)
    response_tag = predict(vector)
    answer = random.choice(responses[response_tag])
    return answer
