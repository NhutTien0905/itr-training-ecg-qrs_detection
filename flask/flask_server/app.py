import json
from io import BytesIO
import base64
import numpy as np
import requests
from flask import Flask, request, jsonify

# from flask_cors import CORS
app = Flask(__name__)

@app.route('/qrs_detection/predict/', methods=['POST'])
def image_classifier():
    # Decoding and pre-processing base64 image
    base64_encoded_data = BytesIO(base64.b64decode(request.form['b64']))
    imgs = np.load(base64_encoded_data)

    data = json.dumps({
        "signature_name": "channels",
        "instances": imgs.tolist()
    })
    
    headers = {"content-type": "application/json"}

    # Making POST request
    json_response = requests.post(
        'http://localhost:8501/v1/models/qrs:predict',
        data=data,
        headers=headers)

    # Decoding results from TensorFlow Serving server
    if json_response.status_code == 200:
        y_pred = json.loads(json_response.text)['predictions']
        # y_pred = np.argmax(y_pred, axis=-1)
        return jsonify(y_pred)
    else:
        return None