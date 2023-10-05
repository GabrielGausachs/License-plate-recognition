"""
Test - Detect license plates in images
"""

import os
import sys
import numpy as np

import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plate_detector import find_plate

def test_with_plates_big_dataset():
    count = 0
    parcial_test = 0
    carpeta_imagenes = "..\\img\\annotations\\images"
    carpeta_xml = "..\\img\\annotations\\annotations"
    for nombre_archivo, nombre__xml in zip(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes)), os.listdir(os.path.join(os.path.dirname(__file__), carpeta_xml))):
        imatge = os.path.join(os.path.dirname(
            __file__), carpeta_imagenes, nombre_archivo)
        xml = os.path.join(os.path.dirname(
            __file__), carpeta_xml, nombre__xml)
        tree = ET.parse(xml)
        root = tree.getroot()
        object_element = root.find('object')
        bndbox_element = object_element.find('bndbox')
        xmin = int(bndbox_element.find('xmin').text)
        ymin = int(bndbox_element.find('ymin').text)
        xmax = int(bndbox_element.find('xmax').text)
        ymax = int(bndbox_element.find('ymax').text)
        try:
            img, box = find_plate(imatge)
            #xmin, ymin, xmax, ymax of box
            min_x, min_y = np.min(box, axis=0)
            max_x, max_y = np.max(box, axis=0)

            if min_x <= xmin and min_y <= ymin and max_x >= xmax and max_y >= ymax:
                count += 1
                print("Test passed")
            else:
                print("Test failed")

            if min_x <= xmin:
                parcial_test += 0.25
            if min_y <= ymin: 
                parcial_test += 0.25
            if max_x >= xmax:
                parcial_test += 0.25
            if max_y >= ymax:
                parcial_test += 0.25

        except:
            print("Erro with image: ", nombre_archivo)
    print("Accuracy: ", count / len(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes))))
    print("There are ", count, " of ", len(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes)))*4, "points inside")


if __name__ == "__main__": 
    test_with_plates_big_dataset()

    

