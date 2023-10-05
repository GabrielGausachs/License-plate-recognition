"""
Test - Identify characters in a license plate using Deep Learning
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_helper import validate_images
from utils.characters_identifier_deep import identify_character


if __name__ == "__main__":
    validate_images(identify_character, False, True)
