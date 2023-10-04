"""
Test - Identify characters in a license plate using Python Library tesseract
"""

import os
import sys

import cv2
import pytesseract
from matplotlib import pyplot as plt
import Levenshtein

from test_helper import test_helper

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.characters_identifier_ocr import identify_character
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


def test_with_plates(print_compare_img=False):
    total_correct_full_plate = 0
    total_correct_segmented_plate = 0
    total_letters = 0
    total = 0
    img_directory = "../img/plates/"
    file_directory = os.path.dirname(os.path.realpath(__file__))

    for test_img in os.listdir(os.path.join(file_directory, img_directory)):
        img_path = os.path.join(file_directory, img_directory + test_img)

        # Find plate
        plate = find_plate(img_path)

        # Segmentate characters
        predicted_all = segmentate_characters("temp_plate.png")

        # Predict characters in a segmented plate
        predicted_full = identify_character(plate)

        # Delete non-alphanumeric characters and whitespace
        predicted_full = ''.join(e for e in predicted_full if e.isalnum())

        # Predict characters in a segmented plate
        predicted_segmentated = []
        for character in predicted_all:
            character_result = identify_character(character).strip()
            predicted_segmentated.append(''.join(e for e in character_result if e.isalnum()))

        # Join segmented characters and remove spaces
        predicted_segmentated = "".join(predicted_segmentated).replace(" ", "")

        # Actual characters
        actual_characters = test_img.split(".")[0]

        # Call test_helper for accuracy calculation and logging
        accuracy_full_plate, accuracy_segmented_plate = test_helper(actual_characters, predicted_full, predicted_segmentated)

        if print_compare_img:
            show_image(plate, "Final")

        # Check if correct for full plate
        if accuracy_full_plate == 100:
            total_correct_full_plate += 1

        # Check if correct for segmented plate
        if accuracy_segmented_plate == 100:
            total_correct_segmented_plate += 1

        total_letters += len(actual_characters)
        total += 1

    # Calculate overall accuracy percentages
    overall_percentage_accuracy_full_plate = (total_correct_full_plate / total) * 100
    overall_percentage_accuracy_segmented_plate = (total_correct_segmented_plate / total) * 100

    # Print overall test results
    print("\n--------------------")
    print(f"Total correct for full plate: {total_correct_full_plate} - {overall_percentage_accuracy_full_plate:.2f}%")
    print(f"Total correct for segmented plate: {total_correct_segmented_plate} - {overall_percentage_accuracy_segmented_plate:.2f}%")

    # Check overall test result based on overall accuracy percentages
    if overall_percentage_accuracy_full_plate >= 90 and overall_percentage_accuracy_segmented_plate >= 90:
        print("\n------------------ Test passed! ------------------")
        return True
    else:
        print("\n------------------ Test failed! ------------------")
        return False


if __name__ == "__main__":
    test_with_plates()
