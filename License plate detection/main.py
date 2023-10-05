import cv2
import numpy as np
import pytesseract as pt
import imutils
import os


def print_image(image, name):
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


for i in os.listdir("License plate detection/Lateral"):

    plate = cv2.imread("License plate detection/Lateral/" + i)

    # Resize image
    plate = imutils.resize(plate, width=1000)

    plate_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

    # Aplicar erode e dilate para remover ruídos varias vezes
    plate_gray = cv2.erode(plate_gray, None, iterations=2)
    plate_gray = cv2.dilate(plate_gray, None, iterations=2)

    print_image(plate_gray, "Plate")

    # Aplicar blur para suavizar a imagem
    plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)

    print_image(plate_gray, "Plate")

    # Encontrar contornos
    cnts = cv2.findContours(
        plate_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Ordenar contornos
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    # Dibujar contorno
    cv2.drawContours(plate, [cnts], -1, (0, 255, 0), 3)

    # Máscara para obter a imagem apenas na área do contorno
    mask = np.zeros(plate_gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [contorno], 0, 255, -1)
    new_image = cv2.bitwise_and(plate, plate, mask=mask)

    # Recortar imagem
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    plate_cropped = plate_gray[topx:bottomx+1, topy:bottomy+1]

    # Mostrar imagem recortada
    cv2.imshow("Plate", plate_cropped)
    cv2.waitKey(0)
