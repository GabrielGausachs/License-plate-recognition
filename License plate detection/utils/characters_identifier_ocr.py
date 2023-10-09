"""
Identify characters in a license plate
using Python Library tesseract
"""

import cv2
import easyocr
from matplotlib import pyplot as plt
import numpy as np


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


def identify_character(bw_img, print_img=False, letter=False):
    """
    Function to identify characters in a license plate

    Args:
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """

    _, thresh = cv2.threshold(bw_img, 127, 255, cv2.THRESH_BINARY)

    # show_image(thresh, "invert")

    reader = easyocr.Reader(lang_list=['en'], gpu=False)

    convert_num2char = {"6": "G", "2": "Z"}
    convert_char2num = {"L": "4", "G": "6", "Z": "2"}

    char = scan_letter(reader=reader, letter=thresh)
    char = char if char else '-'
    if not letter and char.isalpha():
        char = convert_char2num[char]
    elif letter and char.isdigit():
        char = convert_num2char[char]

    if print_img:
        plt.imshow(thresh, cmap="gray")
        plt.show()

    return char
