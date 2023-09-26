"""
Identify characters in a license plate
using Python Library tesseract
"""

import cv2
import pytesseract
import os
from matplotlib import pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def identify_character(img):
    """
    Function to identify characters in a license plate
    
    Args:
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """
    
    # Preprocess image
    img = cv2.GaussianBlur(img, (3,3), 0)
    thresh  = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening
    
    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    
    # print(f"Predicted characters: {data}")
    
    # Show image
    # plt.imshow(invert, cmap='gray')
    # plt.show()
    
    return data
    

def test():
    """
    Function to test the identify_character function with the digits image files

    Args:
        img (string): image file path
    """
    total_correct = 0
    total = 0
    img_directory = '../img/digits/'
    file_directory = os.path.dirname(os.path.realpath(__file__))
    print(file_directory)
    for character in os.listdir(os.path.join(file_directory, img_directory)):
        print("Predicting character: ", character)
        for img in os.listdir(os.path.join(file_directory, img_directory + character)):
            img_path = os.path.join(file_directory, img_directory + character + '/' + img)
            img = cv2.imread(img_path, 0)
            predicted = identify_character(img)
            if predicted == character:
                total_correct += 1
            total += 1
        print('Total correct for {}: {}'.format(character, total_correct))
    print('Accuracy: ', total_correct/total)
    if total_correct/total >= 0.9:
        print('Test passed')
        return True
    else:
        print('Test failed')
        return False


if __name__ == '__main__':
    test()