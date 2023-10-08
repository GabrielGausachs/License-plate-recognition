"""
Identify characters in a license plate
using Python Library tesseract
"""

import cv2
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


def identify_character(model_name, bw_img):
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
    elif model_name == "model_cnn_2":
        model_to_use = model_cnn_2
    elif model_name == "model_nn":
        model_to_use = model_nn
    else:
        raise Exception("Model not found")
    
    

    # Predict character - original image 1x1032
    data = model_to_use.predict(thresh.reshape(40, 32, 1))
    data = data.argmax()
    data = chr(data + 65)

    return data
