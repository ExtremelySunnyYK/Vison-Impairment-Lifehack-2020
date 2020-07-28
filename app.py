import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

#Keras Models
import keras
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import backend as K
from keras.activations import elu
from keras.optimizers import Adam
from keras.models import Sequential
from keras.engine import Layer, InputSpec
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Conv2D, Flatten, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import cohen_kappa_score
from keras_radam import RAdam
from keras_contrib.layers import GroupNormalization



# Some utilites
import numpy as np
from util import base64_to_pil
# from setup import preprocess_image
import setup




# Declare a flask app
app = Flask(__name__)
MODEL_PATH = 'models/effnet_b5_model.h5'


# You can use pretrained model from Keras
# Check https://keras.io/applications/
from efficientnet.keras import EfficientNetB5 as EfficientNet

EFFNET = 5

effnet_to_img_size = dict(enumerate(
        [(224, 224), (240, 240), (260, 260), (300, 300), (380, 380), (456, 456), (528, 528), (600, 600)]
    ))
IMG_WIDTH, IMG_HEIGHT = effnet_to_img_size[EFFNET]
CHANNELS = 3
# Specify number of epochs
effnet_to_nb_epochs = dict(enumerate(
    [30, 30, 30, 30, 30, 25, 20, 15]
    # [30, 22, 17, 9, 8, 8]
))
EPOCHS = effnet_to_nb_epochs[EFFNET]
BATCH_SIZE = 4


effnet = EfficientNet(weights=None,  # None,  # 'imagenet',
                            include_top=False,
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))

# Specify number of epochs
effnet_to_nb_epochs = dict(enumerate(
    [30, 30, 30, 30, 30, 25, 20, 15]
    # [30, 22, 17, 9, 8, 8]
))

for i, layer in enumerate(effnet.layers):
        if "batch_normalization" in layer.name:
            effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)

# Initialize model
print("> Building Model ...")
model = Sequential()
model.add(effnet)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(5, activation=elu))
model.add(Dense(1, activation="linear"))
model.compile(loss='mse',
                optimizer=Adam(learning_rate=0.00005), 
                metrics=['mse', 'acc'])

model.load_weights("models/effnet_b5_model.h5")

print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()

# Load your own trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary


def model_predict(img, model):
    """
    :param img : Image posted by user
    :param model : Efficentnet

    :return: Image Diagnosis
    """

    # img = img.resize((224, 224))
    # Preprocessing the image
    img_arr = keras.preprocessing.image.img_to_array(img)
    img_arr = setup.preprocess_image(img_arr,IMG_WIDTH,IMG_HEIGHT)
    rescale=1 / 255
    img_arr *= rescale
    img_arr = np.array([img_arr])
    predictions = model.predict(img_arr)
    print(predictions)

    coefficients = [0.51,1.51,2.52,3.52]
    optR = setup.OptimizedRounder()
    img_diag = optR.predict(predictions, coefficients).astype(np.uint8)
    print(img_diag)

    print("> Prediction Finished")

    return img_diag


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

def severity(sev_num):
    """
    :parameter sev_num : Severity number

    :returns sev_diag : Severity of diagnosis
    """
    sev_dict = {
        "0":"Level 0 - No DR",
    "1":"Level 1 - Mild",
    "2":"Level 2 - Moderate",
    "3":"Level 3 - Severe",
    "4":"Level 4 - Proliferative Diabetic Retinopathy",
    }

    sev_diag = sev_dict[sev_num]

    return sev_diag

def serve_pil_image(pil_img):
    img_io = StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        # ben_img,input_img,heated_img = setup.visualize(img)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        print(preds)
        score = str(preds[0][0])
        print(score)
        result = severity(score)
        

        # Serialize the result, you can add additional fields
        return jsonify(result=result)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
