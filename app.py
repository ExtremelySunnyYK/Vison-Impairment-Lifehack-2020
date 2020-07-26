import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
from predict import *

app = Flask(__name__)


def get_score(left_eye,right_eye):
    """
    Returns AREDS Score given image
    :Parameters : Image of eyes
    
    :returns : AREDS Score
    """"
    score = predict_simplified_score.flask_predict()
    print("* Model loaded!")

    return score

@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    left_eye = Image.open(io.BytesIO(decoded))
    right_eye = Image.open(io.BytesIO(decoded))
    print(" * Loading Keras model...")
    score = get_score(left_eye,right_eye)
    # can add other features as well
    # such as drusen size ...

    response = {
        'AREDS': {
            'Level': score,
        }
    }
    return jsonify(response)