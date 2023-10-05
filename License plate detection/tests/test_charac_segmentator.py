"""
Test - Segmentate characters in license plates
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ..utils.characters_segmentator import segmentate_characters

if __name__ == "__main__":
    segmentate_characters("plate_test.png")
