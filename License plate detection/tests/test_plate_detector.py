"""
Test - Detect license plates in images
"""

import os
import sys
import numpy as np
import cv2

import xml.etree.ElementTree as ET

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.plate_detector import find_plate


def test_with_plates_big_dataset():
    count = 0
    parcial_test = 0
    carpeta_imagenes = "..\\img\\annotations\\images"
    carpeta_xml = "..\\img\\annotations\\annotations"

    image_files = sorted(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes)))
    annotations_files = sorted(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_xml)))

    for nombre_archivo, nombre__xml in zip(image_files, annotations_files):
        print("----------------------------------")
        print(f"Image: {nombre_archivo}")
        print(f"XML: {nombre__xml}\n")
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
            img, box = find_plate(imatge)  # Asume que find_plate está definida correctamente

            # xmin, ymin, xmax, ymax of box
            min_x, min_y = np.min(box, axis=0)
            max_x, max_y = np.max(box, axis=0)

            # Redondear las coordenadas a números enteros
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            min_x = int(min_x)
            min_y = int(min_y)
            max_x = int(max_x)
            max_y = int(max_y)

            # Print all the values
            print("Real - xmin: ", xmin, " ymin: ", ymin, " xmax: ", xmax, " ymax: ", ymax)
            print("Predicted - min_x: ", min_x, " min_y: ", min_y, " max_x: ", max_x, " max_y: ", max_y)

            imatge = cv2.imread(imatge)

            # Dibujar las cajas en la imagen
            cv2.rectangle(imatge, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(imatge, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

            # Mostrar la imagen con ambas cajas dibujadas
            cv2.imshow("Imagen con cajas", imatge)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

        except Exception as e:
            print(f"Error with image {nombre_archivo}: {e}")
    print("Accuracy: ", count /
          len(os.listdir(os.path.join(os.path.dirname(__file__), carpeta_imagenes))))
    print("There are ", count, " of ", len(os.listdir(os.path.join(
        os.path.dirname(__file__), carpeta_imagenes)))*4, "points inside")


if __name__ == "__main__":
    test_with_plates_big_dataset()
