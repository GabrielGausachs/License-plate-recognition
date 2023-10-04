"""
Test - Identify characters in a license plate using Python Library tesseract
"""

import os
import sys

import cv2
import pytesseract
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.characters_identifier_tessereact import identify_character
from utils.characters_segmentator import segmentate_characters
from utils.plate_detector import find_plate

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


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


def test_with_plates():
    """
    Function to test the identify_character function with the digits image files

    Args:
        img (string): image file path
    """

    total_correct = 0
    total = 0
    img_directory = "../img/plates/"
    file_directory = os.path.dirname(os.path.realpath(__file__))
    print(file_directory)

    for test_img in os.listdir(os.path.join(file_directory, img_directory)):
        img_path = os.path.join(file_directory, img_directory + test_img)

        # Find plate
        plate = find_plate(img_path)

        # Segmentate characters
        predicted_all = segmentate_characters("temp_plate.png")

        # Predict characters in a segmented plate
        predicted_full = identify_character(plate)

        # Delete not alphanumeric characters
        predicted_full = ''.join(e for e in predicted_full if e.isalnum())

        # Predict characters in a segmented plate
        predicted_segmentated = []
        for character in predicted_all:
            predicted_segmentated.append(
                identify_character(character).strip())

        # Show image and predicted characters
        print("\n--------------------")
        print(f'Actual characters: {test_img.split(".")[0]}')
        print(f"Predicted with full plate: {predicted_full}")
        print("Predicted with segmented plate: ",
              "".join(predicted_segmentated))
        show_image(plate, "Final")

        # Check if correct
        if predicted_full == test_img.split('.')[0] or "".join(predicted_segmentated) == test_img.split('.')[0]:
            total_correct += 1
            print(
                f'Correct! - {test_img.split(".")[0]} - {predicted_full} - {"".join(predicted_segmentated)}')
        else:
            print(
                f'Incorrect! - {test_img.split(".")[0]} - {predicted_full} - {"".join(predicted_segmentated)}')
        total += 1

    # Check results
    print("Accuracy: ", total_correct / total)
    if total_correct / total >= 0.9:
        print("Test passed")
        return True
    else:
        print("Test failed")
        return False


if __name__ == "__main__":
    test_with_plates()
