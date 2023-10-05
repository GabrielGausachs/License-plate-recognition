"""
Test - Segmentate characters in license plates
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.characters_segmentator import segmentate_characters
from utils.plate_detector import find_plate

if __name__ == "__main__":
    for img in os.listdir(os.path.join(os.path.dirname(__file__), "../img/plates")):
        print("------------------------------")
        print("Predicting plate: ", img)
        find_plate(os.path.join(os.path.dirname(__file__), "../img/plates", img))
        characters = segmentate_characters()
        if len(characters) == 7:
            print(f"Test passed - {img}\n")
        else:
            print(f"------------------Test failed - {img} - {len(characters)}------------------\n")
