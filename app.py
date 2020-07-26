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
from examples import predict_simplified_score

app = Flask(__name__)


def get_score(left_eye,right_eye):
    score = predict_simplified_score.flask_predict()
    print("* Model loaded!")

    return score