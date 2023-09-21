# Plate Detection with OpenCV and Python

import cv2
import numpy as np
import pytesseract

# Lee la imagen
img = cv2.imread("License plate detection\img1.jpg")

# Convierte la imagen a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica un filtro Gaussiano para suavizar la imagen
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Aplica Canny para detectar los bordes
canny = cv2.Canny(blur, 100, 200)

# Encuentra contornos en la imagen
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Dibujar los contornos en la imagen
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

# Filtrar los contornos que podrían ser matrículas
candidate_contours = []
for contour in contours:
        # Mirar que el contorno tenga cuatro vértices
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
        # Mirar que el contorno tenga un área mínima
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > 200:
            candidate_contours.append(contour)

# Dibujar los contornos candidatos en la imagen
cv2.drawContours(img, candidate_contours, -1, (0, 255, 0), 2)

# Buscar matrículas en los candidatos
for contour in candidate_contours:
    x, y, w, h = cv2.boundingRect(contour)
    roi = gray[y:y + h, x:x + w]
    #text = pytesseract.image_to_string(roi, config='--psm 8')

    # # Comprueba si el texto contiene caracteres alfanuméricos y es lo suficientemente largo
    # if text.isalnum() and len(text) > 3:
    #     print("Matrícula detectada:", text)
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Muestra la imagen con las matrículas detectadas
cv2.imshow("Matrículas Detectadas", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


