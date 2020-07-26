import os
import sys

# Standard dependencies
import cv2
import time
import scipy as sp
import numpy as np
import random as rn
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

# Machine Learning
import tensorflow as tf
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

from sklearn.metrics import accuracy_score


"""
Get predictions and labels from the generator

:param model: A Keras model object
:param generator: A Keras ImageDataGenerator object

:return: A tuple with two Numpy Arrays. One containing the predictions
and one containing the labels
"""



def setup():
    EFFNET = 5
    exec('from efficientnet.keras import EfficientNetB{} as EfficientNet'.format(EFFNET))

    # Path specifications
    KAGGLE_DIR = 'data/'
    TRAIN_DF_PATH = KAGGLE_DIR + "train.csv"
    TEST_DF_PATH = KAGGLE_DIR + 'test.csv'
    TRAIN_IMG_PATH = KAGGLE_DIR + "train_images/"
    TEST_IMG_PATH = KAGGLE_DIR + 'test_images/'

    # Specify title of our final model
    SAVED_MODEL_NAME = 'notebook/effnet_b5_model.h5'

    # Set seed for reproducability
    seed = 1234
    rn.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # For keeping time. GPU limit for this competition is set to ± 9 hours.
    t_start = time.time()

    TRAIN_TEST_RATIO = 1 - (70 + 15) / (70 + 15 + 15)
    TRAIN_VAL_RATIO = 1 - 70 / (70 + 15)

    print("Image IDs and Labels (TRAIN)")
    train_df = pd.read_csv(TRAIN_DF_PATH).sample(frac=1, random_state=seed)
    # Add extension to id_code
    train_df['id_code'] = train_df['id_code'] + ".png"
    print(f"Training images: {train_df.shape[0]}")
    # display(train_df.head())

    print("Image IDs (TEST)")
    test_df = pd.read_csv(TEST_DF_PATH)  # .sample(frac=1, random_state=seed)
    # Add extension to id_code
    test_df['id_code'] = test_df['id_code'] + ".png"
    test_df = test_df.rename(columns={'diagnosis': 'label'})
    print(f"Testing Images: {test_df.shape[0]}")
    # display(test_df.head())

    # Specify image size
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


def data_preprocessing():
    # Example of preprocessed images from every label
    fig, ax = plt.subplots(1, 5, figsize=(15, 6))
    for i in range(5):
        sample = train_df[train_df['diagnosis'] == i].sample(1)
        image_name = sample['id_code'].item()
        X = preprocess_image(cv2.imread(f"{TRAIN_IMG_PATH}{image_name}"))
        ax[i].set_title(f"Image: {image_name}\n Label = {sample['diagnosis'].item()}", 
                        weight='bold', fontsize=10)
        ax[i].axis('off')
        ax[i].imshow(X);
    # Labels for training data
    # y_labels = train_df['diagnosis'].values

def modelling():
    # We use a small batch size so we can handle large images easily
    BATCH_SIZE = 4

    # Add Image augmentation to our generator
    train_datagen = ImageDataGenerator(rotation_range=360,
                                    horizontal_flip=True,
                                    vertical_flip=True,
                                    validation_split=TRAIN_VAL_RATIO,
                                    preprocessing_function=preprocess_image, 
                                    rescale=1 / 255.)

    # Use the dataframe to define train and validation generators
    train_generator = train_datagen.flow_from_dataframe(train_df, 
                                                        x_col='id_code', 
                                                        y_col='diagnosis',
                                                        directory = TRAIN_IMG_PATH,
                                                        target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                        batch_size=BATCH_SIZE,
                                                        class_mode='other', 
                                                        subset='training',
                                                        seed=seed)

    val_generator = train_datagen.flow_from_dataframe(train_df, 
                                                    x_col='id_code', 
                                                    y_col='diagnosis',
                                                    directory = TRAIN_IMG_PATH,
                                                    target_size=(IMG_WIDTH, IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='other',
                                                    subset='validation',
                                                    seed=seed)
        # Load in EfficientNetB5
    effnet = EfficientNet(weights=None,  # None,  # 'imagenet',
                            include_top=False,
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    effnet.load_weights(
        'notebook/effnet_b5_model.h5'
        )
    )

    # Replace all Batch Normalization layers by Group Normalization layers
    for i, layer in enumerate(effnet.layers):
        if "batch_normalization" in layer.name:
            effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
            
def build_model():
    """
    A custom implementation of EfficientNetB5
    for the APTOS 2019 competition
    (Regression)
    """
    model = Sequential()
    model.add(effnet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(5, activation=elu))
    model.add(Dense(1, activation="linear"))
    model.compile(loss='mse',
                  optimizer=RAdam(learning_rate=0.00005), 
                  metrics=['mse', 'acc'])
    print(model.summary())
    return model



def get_preds_and_labels(model, generator):
    """
    Get predictions and labels from the generator
    
    :param model: A Keras model object
    :param generator: A Keras ImageDataGenerator object
    
    :return: A tuple with two Numpy Arrays. One containing the predictions
    and one containing the labels
    """
    preds = []
    labels = []
    for _ in range(int(np.ceil(generator.samples / BATCH_SIZE))):
        x, y = next(generator)
        preds.append(model.predict(x))
        labels.append(y)
    # Flatten list of numpy arrays
    return np.concatenate(preds).ravel(), np.concatenate(labels).ravel().astype(int)

def crop_image_from_gray(img, tol=7):
    """
    Applies masks to the orignal image and 
    returns the a preprocessed image with 
    3 channels
    
    :param img: A NumPy Array that will be cropped
    :param tol: The tolerance used for masking
    
    :return: A NumPy array containing the cropped image
    """
    # If for some reason we only have two channels
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    # If we have a normal RGB images
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def preprocess_image(image, sigmaX=10):
    """
    The whole preprocessing pipeline:
    1. Read in image
    2. Apply masks
    3. Resize image to desired size
    4. Add Gaussian noise to increase Robustness
    
    :param img: A NumPy Array that will be cropped
    :param sigmaX: Value used for add GaussianBlur to the image
    
    :return: A NumPy array containing the preprocessed image
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    # image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,sigmaX), -4, 128)
    return image


class Metrics(Callback):
    """
    A custom Keras callback for saving the best model
    according to the Quadratic Weighted Kappa (QWK) metric
    """
    def on_train_begin(self, logs={}):
        """
        Initialize list of QWK scores on validation data
        """
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        """
        Gets QWK score on the validation data
        
        :param epoch: The current epoch number
        """
        # Get predictions and convert to integers
        y_pred, labels = get_preds_and_labels(model, val_generator)
        y_pred = np.rint(y_pred).astype(np.uint8).clip(0, 4)
        # We can use sklearns implementation of QWK straight out of the box
        # as long as we specify weights as 'quadratic'
        _val_kappa = cohen_kappa_score(labels, y_pred, weights='quadratic')
        self.val_kappas.append(_val_kappa)
        print(f"val_kappa: {round(_val_kappa, 4)}")

        # if _val_kappa == max(self.val_kappas):
        #     print("Validation Kappa has improved. Saving model.")
        #     self.model.save(SAVED_MODEL_NAME)
        return


class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize Quadratic Weighted Kappa score
    """
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p.astype(int)

    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']


def main():
    """"
    Get predictions and labels from the model

    :param image: image of the upload

    :returns: ARED's Score
    """
    setup()

    # change code for 1 image
    for i in range(5):
        sample = train_df[train_df['diagnosis'] == i].sample(1)
        image_name = sample['id_code'].item()
        X = preprocess_image(cv2.imread(f"{TRAIN_IMG_PATH}{image_name}"))
        ax[i].set_title(f"Image: {image_name}\n Label = {sample['diagnosis'].item()}", 
                        weight='bold', fontsize=10)
        ax[i].axis('off')
        ax[i].imshow(X);

     # We use a small batch size so we can handle large images easily
    BATCH_SIZE = 4

   
        # Load in EfficientNetB5
    effnet = EfficientNet(weights=None,  # None,  # 'imagenet',
                            include_top=False,
                            input_shape=(IMG_WIDTH, IMG_HEIGHT, CHANNELS))
    effnet.load_weights(
        'notebook/effnet_b5_model.h5'
        )
    )

    # Replace all Batch Normalization layers by Group Normalization layers
    for i, layer in enumerate(effnet.layers):
        if "batch_normalization" in layer.name:
            effnet.layers[i] = GroupNormalization(groups=32, axis=-1, epsilon=0.00001)
     

    # Initialize model
    model = build_model()
    
    # For tracking Quadratic Weighted Kappa score
    kappa_metrics = Metrics()
    # Monitor MSE to avoid overfitting and save best model
    es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=EPOCHS // 2)
    rlr = ReduceLROnPlateau(monitor='val_loss', 
                            factor=0.5, 
                            patience=EPOCHS // 4, 
                            verbose=1, 
                            mode='auto', 
                            epsilon=0.0001)

    # Optimize on validation data and evaluate again
    preds = model.predict(x)


   

if __name__=="main":
    main()


