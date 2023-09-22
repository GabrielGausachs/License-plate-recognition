

# provar de fer la dilatacio abans que la binarització
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os

carpeta_imagenes = "C:/Users/arnau/OneDrive/Escritorio/ARNAU G/UNI/4t ENG DADES/PSIV II/Lateral"

for nombre_archivo in os.listdir(carpeta_imagenes):
    ruta_completa = os.path.join(carpeta_imagenes, nombre_archivo)
    image = cv2.imread(ruta_completa)
    image = imutils.resize(image, width=500)
    image_copy = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    cv2.imshow("blackhat Image", blackhat)
    cv2.waitKey(0)
    
    kernel = np.ones((1,4), np.uint8)
    erode_img3 = cv2.erode(image, kernel) 
    cv2.imshow('erode1', erode_img3)
    cv2.waitKey(0)

    kernel = np.ones((10, 23), np.uint8) 
    dilate_image = cv2.dilate(blackhat, kernel, iterations=1)
    cv2.imshow('dilate1', dilate_image)
    cv2.waitKey(0)
    """
    kernel = np.ones((1, 4), np.uint8)
    erode_img = cv2.erode(image, kernel) 
    cv2.imshow('erode1', erode_img)
    cv2.waitKey(0)
    """
    kernel = np.ones((10, 23), np.uint8) 
    dilate_image_2 = cv2.dilate(blackhat, kernel, iterations=1)
    cv2.imshow('dilate2', dilate_image_2)
    cv2.waitKey(0)
    """
    kernel = np.ones((1,4), np.uint8)
    erode_img2 = cv2.erode(image, kernel) 
    cv2.imshow('erode1', erode_img2)
    cv2.waitKey(0)
    """
    kernel = np.ones((10, 23), np.uint8) 
    dilate_image_3 = cv2.dilate(blackhat, kernel, iterations=1)
    cv2.imshow('dilate2', dilate_image_3)
    #cv2.waitKey(0)
 

    _, binary_image = cv2.threshold(dilate_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary Image", binary_image)
    cv2.waitKey(0)

    contours=cv2.findContours(binary_image.copy(),cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours,key=cv2.contourArea, reverse = True)[:10]
    screenCnt = None
    
    area = 0

    for contour in contours:
        # Aproximar el contorno
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Si el contorno aproximado tiene 4 vértices, puede ser una matrícula
        if  int(cv2.contourArea(contour))>area and cv2.contourArea(contour)<8000:  #len(approx) == 4 and
            area = int(cv2.contourArea(contour))
            # Dibujar un rectángulo alrededor de la matrícula
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            screenCnt = approx
            break

    cv2.imshow('box', cv2.rectangle(image_copy, tuple(box[0].astype(int)), tuple(box[2].astype(int)),(0,0,255),2))
    cv2.waitKey(0)