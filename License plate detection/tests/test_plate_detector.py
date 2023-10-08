"""
Test - Detect license plates in images
"""

import os
import sys
import numpy as np
import cv2
import imutils

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
        image = os.path.join(os.path.dirname(
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
            img, box = find_plate(image)  # Asume que find_plate está definida correctamente

            for i in range(len(box)):
                if i == 0:
                    box[i][0] -= 30  # Disminuir x en -30 unidades
                    box[i][1] -= 30  # Aumentar y en -30 unidades
                elif i == 1:
                    box[i][0] += 30  # Aumentar x en -30 unidades
                    box[i][1] -= 30  # Aumentar y en -30 unidades
                elif i == 2:
                    box[i][0] += 30  # Aumentar x en -30 unidades
                    box[i][1] += 30  # Disminuir y en -30 unidades
                elif i == 3:
                    box[i][0] -= 30  # Disminuir x en -30 unidades
                    box[i][1] += 30  # Disminuir y en -30 unidades

            # xmin, ymin, xmax, ymax of box
            print(box)
            max_x = max(box[0][0], box[1][0], box[2][0], box[3][0])
            min_x = min(box[0][0], box[1][0], box[2][0], box[3][0])
            max_y = max(box[0][1], box[1][1], box[2][1], box[3][1])
            min_y = min(box[0][1], box[1][1], box[2][1], box[3][1])

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

            image = cv2.imread(image)

            # Dibujar las cajas en la imagen
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            image_2 = imutils.resize(image, width=500)
            cv2.rectangle(image_2, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)

            # Mostrar la imagen con ambas cajas dibujadas
            # cv2.imshow("Imagen con cajas", image_2)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            factor = image.shape[0] / image_2.shape[0]

            if min_x*factor <= xmin and min_y*factor <= ymin and max_x*factor >= xmax and max_y*factor >= ymax:
                count += 1
                print("\nTest passed\n")
            else:
                print("\nTest failed\n")

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
        os.path.dirname(__file__), carpeta_imagenes))), "points inside")


if __name__ == "__main__":
    test_with_plates_big_dataset()
