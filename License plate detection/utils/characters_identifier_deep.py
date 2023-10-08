"""
Identify characters in a license plate
using Python Library tesseract
"""
import numpy as np
import cv2
import os

from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import os

# Load all saved h5 models
model_cnn_1 = load_model(os.path.join(os.path.dirname(
    __file__), "saved_models/model_cnn.h5"))
model_cnn_2 = load_model(os.path.join(os.path.dirname(
    __file__), "saved_models/model_cnn_2.h5"))
model_nn = load_model(os.path.join(os.path.dirname(
    __file__), "saved_models/model_nn.h5"))


def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def identify_character(bw_img, model_name= 'model_nn'):
    """
    Function to identify characters in a license plate using a Deep Learning model

    Args:
        model_name: name of the model to be used
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """

    # Preprocess image
    img = cv2.GaussianBlur(bw_img, (3, 3), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if model_name == "model_cnn":
        model_to_use = model_cnn_1
        resized_image = cv2.resize(thresh, (32, 40))
        resized_image = resized_image.reshape(1, 40, 32, 1)
    elif model_name == "model_cnn_2":
        model_to_use = model_cnn_2
        #We need shape (1,96,96,3)
        resized_image = cv2.resize(thresh, (32, 40))
        resized_image = resized_image.reshape(1, 40, 32, 1)
    elif model_name == "model_nn":
        resized_image = cv2.resize(thresh, (32, 40))
        resized_image = resized_image.reshape(1, 40, 32, 1)
        model_to_use = model_nn
    else:
        raise Exception("Model not found")
    
    



    data = model_to_use.predict(resized_image)
    data = data.argmax()
    data = chr(data + 65)

    return data
