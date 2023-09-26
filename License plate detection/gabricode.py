

# provar de fer la dilatacio abans que la binarització
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os

config = {
    "print": {
        "original": False,
        "gray": False,
        "blackhat": True,
        "step": False,
        "binary": False,
        "final": True
    }
}


def show_image(image, title="Image"):
    if config["print"][title.lower()]:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        
def dilate(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iters)


def erode(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iters)


carpeta_imagenes = "Lateral"

i = 0

for nombre_archivo in os.listdir(os.path.join(os.getcwd(), carpeta_imagenes)):
    if i == 1:
        break
    ruta_completa = os.path.join(carpeta_imagenes, nombre_archivo)
    image = cv2.imread(ruta_completa)
    show_image(image, "Original")
    image = imutils.resize(image, width=1000)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray, "Gray")
    
    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)
    
    show_image(blackhat, "Blackhat")

    blackhat = erode(blackhat, (2, 5), 20)
    blackhat = dilate(blackhat, (5, 20), 50)
    blackhat = erode(blackhat, (20, 5), 50)
    blackhat = erode(blackhat, (10, 5), 20)
    
    blackhat = dilate(blackhat, (10, 5), 60)
    blackhat = dilate(blackhat, (10, 5), 60)
    blackhat = dilate(blackhat, (5, 10), 60)
    blackhat = dilate(blackhat, (10, 30), 30)
    
    blackhat = dilate(blackhat, (2, 10), 30)
    blackhat = dilate(blackhat, (2, 10), 25)
    blackhat = dilate(blackhat, (2, 10), 15)
    
    # Show image
    show_image(blackhat, "Blackhat")

    _, binary_image = cv2.threshold(blackhat, 127, 255, cv2.THRESH_BINARY)
    show_image(binary_image, "Binary")

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
        if int(cv2.contourArea(contour))>area:
            area = int(cv2.contourArea(contour))
            # Dibujar un rectángulo alrededor de la matrícula
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            screenCnt = approx
            break

    # Draw the contours
    image_final = cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 3)
    show_image(image_final, "Final")
    
    i += 1