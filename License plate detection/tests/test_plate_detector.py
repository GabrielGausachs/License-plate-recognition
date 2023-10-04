"""
Test - Detect license plates in images
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.plate_detector import find_plate

if __name__ == "__main__":
    carpeta_imagenes = "../img/plates"

    for nombre_archivo in os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes)):
        ruta_completa = os.path.join(os.path.dirname(
            __file__), carpeta_imagenes, nombre_archivo)
        find_plate(ruta_completa)
