"""
Test - Detect license plates in images
"""

from utils.plate_detector import find_plate
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_with_plates():
    pass


if __name__ == "__main__":
    carpeta_imagenes = "../img/plates"

    for nombre_archivo in os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes)):
        ruta_completa = os.path.join(os.path.dirname(__file__), carpeta_imagenes, nombre_archivo)
        find_plate(ruta_completa)
