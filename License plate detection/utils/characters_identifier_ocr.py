"""
Identify characters in a license plate
using Python Library tesseract
"""

import easyocr
from matplotlib import pyplot as plt
import os
import cv2


def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


def scan_letter(reader, letter):
    detected_character = None
    results = reader.readtext(letter)
    if results:
        detected_character = results[0][1]
        print("Detected character:", detected_character)
    else:
        print("No character detected.")
    return detected_character


def identify_character(bw_img, print_img=False, position=None):
    """
    Function to identify characters in a license plate

    Args:
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """

    if position:
        if position > 3:
            letter = True
        else:
            letter = False
    else:
        letter = False

    # show_image(bw_img, "Original image")

    reader = easyocr.Reader(lang_list=['en'], gpu=False)

    convert_num2char = {"6": "G", "2": "Z", "0": "D", "4": "H", "8": "B"}
    convert_char2num = {"L": "4", "G": "6", "Z": "2", "D": "0", "H": "4"}

    char = scan_letter(reader=reader, letter=bw_img)
    char = char if char else '-'
    try:
        if not letter and char.isalpha():
            char = convert_char2num[char.upper()]
        elif letter and char.isdigit():
            char = convert_num2char[char]
    except KeyError:
        pass
    if letter and char == "-":
        char = "J"
    elif not letter and char == "-":
        char = "1"

    # if print_img:
    #     plt.imshow(bw_img, cmap="gray")
    #     plt.show()

    return char.upper()


if __name__ == '__main__':
    # Read image in B&W format
    img = cv2.imread(os.path.join(os.path.dirname(os.path.realpath(__file__)), "4.png"), 0)
    # Identify character
    result = identify_character(img, True, False)
