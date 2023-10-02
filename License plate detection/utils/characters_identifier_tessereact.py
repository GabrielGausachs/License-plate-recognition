"""
Identify characters in a license plate
using Python Library tesseract
"""

import cv2
import pytesseract
import os
from matplotlib import pyplot as plt
from plate_detector import find_plate
from characters_segmentator import segmentate_characters

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def identify_character(img, print_img=False):
    """
    Function to identify characters in a license plate

    Args:
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Preprocess image
    img = cv2.GaussianBlur(img, (3, 3), 0)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang="eng", config="--psm 6")

    if print_img:
        print(f"Predicted characters: {data}")
        plt.imshow(invert, cmap="gray")
        plt.show()

    return data


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
        predicted_all = segmentate_characters(plate)

        # Predict characters in a segmented plate
        predicted_segmentated = identify_character(plate)

        # Show image and predicted characters
        print(f'Actual characters: {test_img.split(".")[0]}')
        print(f"Predicted with full plate: {predicted_all}")
        print("Predicted with segmented plate:")
        for character in predicted_segmentated:
            print(character)
        show_image(plate, "Final")

        # Check if correct
        # if predicted == test_img.split('.')[0]:
        #     total_correct += 1
        #     print(f'Correct! - {test_img} - {predicted}')
        # else:
        #     print(f'Incorrect! - {test_img} - {predicted}')
        # total += 1

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

    # Read image
    image = cv2.imread(os.path.join(os.path.dirname(__file__), "test_plate.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # borrar todo lo que no esté en un fondo blanco
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find plate
    data = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

    print(data)
    show_image(image, "Original")

    image = cv2.imread(os.path.join(os.path.dirname(__file__), "C.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # borrar todo lo que no esté en un fondo blanco
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find plate
    data = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

    print(data)
    show_image(image, "Original")

    image = cv2.imread(os.path.join(os.path.dirname(__file__), "4.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # borrar todo lo que no esté en un fondo blanco
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Find plate
    data = pytesseract.image_to_string(image, lang="eng", config="--psm 6")

    print(data)
    show_image(image, "Original")
