from flask import Flask, request, jsonify
import requests
import pickle
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)  

with open('model.pkl', 'rb') as file:
    pipeline = pickle.load(file)

# Hugging Face API details
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}

def query_huggingface_model(text):
    payload = {"inputs": f"give remedies of {text} in summary and in numbered points"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()[0]['generated_text']

@app.route('/', methods=['POST'])
@cross_origin()  

def predict():
    data = request.get_json(force=True)

    if 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request"}), 400

    # Make prediction using the loaded pipeline
    prediction = pipeline.predict([data['text']])

    # Query Hugging Face model with the predicted text
    huggingface_response = query_huggingface_model(prediction[0])

    # Send back the prediction and Hugging Face response
    return jsonify({'prediction': prediction[0], 'huggingface_response': huggingface_response})

@app.route('/hello', methods=['GET'])
@cross_origin()  
def hello():
    return jsonify({'message': 'Hello, World!'})

