
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import model_from_json
import re

model = load_model("D:\\PROJECT\\cyber_bullyingvi\\cyber_bullyingvi\\cyber_bullying\\BI LSTM\\models\\model.keras") 
with open("D:\\PROJECT\\cyber_bullyingvi\\cyber_bullyingvi\\cyber_bullying\\BI LSTM\\models\\model.pkl", 'rb') as f:
    loaded_pickle_data = pickle.load(f)

model = model_from_json(loaded_pickle_data['model'])
model.set_weights(loaded_pickle_data['weights'])

tokenizer = loaded_pickle_data['tokenizer']
label_encoder = loaded_pickle_data['label_encoder']

MAX_SEQUENCE_LENGTH = 100  
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text) 
    text = re.sub(r"@\w+", "", text)  
    text = text.lower()  
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  
    text = re.sub(r"\s+", " ", text).strip() 
    return text

app = Flask(__name__)
CORS(app)
@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', None)
    text= clean_text(text)



    if text is None:
        return jsonify({"error": "No text provided"}), 400

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = model.predict(padded_sequence)
    response = {
        "text": text,
        "prediction": prediction.tolist()  
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)

