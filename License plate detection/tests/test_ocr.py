"""
Test - Identify characters in a license plate using Python Library tesseract
"""

import os
import sys
import cv2
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_helper import validate_images
from utils.characters_identifier_ocr import identify_character


def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def test_with_letter_dataset():
    """
    Function to test the identify_character function with the digits image files

    Args:
        img (string): image file path
    """

    total_correct = 0
    total = 0
    img_directory = "../img/digits/"
    file_directory = os.path.dirname(os.path.realpath(__file__))

    for character in os.listdir(os.path.join(file_directory, img_directory)):
        print("Predicting character: ", character)
        for img in os.listdir(os.path.join(file_directory, img_directory + character)):
            img_path = os.path.join(
                file_directory, img_directory + character + "/" + img
            )
            img = cv2.imread(img_path, 0)
            predicted = identify_character(img)
            if predicted == character:
                total_correct += 1
            total += 1
        print("Total correct for {}: {}".format(character, total_correct))

    print("Accuracy: ", total_correct / total)
    if total_correct / total >= 0.9:
        print("Test passed")
        return True
    else:
        print("Test failed")
        return False


if __name__ == "__main__":
    validate_images(identify_character, False, False)
