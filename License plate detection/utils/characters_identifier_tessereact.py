"""
Identify characters in a license plate
using Python Library tesseract
"""

import cv2
import pytesseract
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()


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
            custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        else:
            custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'
    else:
        custom_config = r'--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789'

    # Perform text extraction
    data = pytesseract.image_to_string(bw_img, config=custom_config)

    # if print_img:
    #     print(f"Predicted characters: {data}")
    #     plt.imshow(bw_img, cmap="gray")
    #     plt.show()

    return data
