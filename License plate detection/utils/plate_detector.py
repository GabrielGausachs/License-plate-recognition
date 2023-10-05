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
        "blackhat": False,
        "dilate": False,
        "binary": False,
        "final": False,
    }
}


def show_image(image, title="Image"):
    if config["print"][title.lower()]:
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()


def dilate(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iterations=iters)


def erode(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iterations=iters)


def find_plate(img_route): 
    image = cv2.imread(img_route)
    show_image(image, "Original")
    image = imutils.resize(image, width=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(gray, "Gray")

    erode_img = erode(gray, (1, 4), 1)
    erode_img = erode(erode_img, (4, 1), 1)

    rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(erode_img, cv2.MORPH_BLACKHAT, rectKern)

    show_image(blackhat, "Blackhat")

    dilate_img = dilate(blackhat, (10, 23), 1)

    dilate_img += 20

    # Show image
    show_image(dilate_img, "Dilate")

    _, binary_image = cv2.threshold(dilate_img, 100, 255, cv2.THRESH_BINARY)
    show_image(binary_image, "Binary")

    contours = cv2.findContours(
        binary_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    area = 0

    for contour in contours:
        # Si el contorno aproximado tiene 4 vértices, puede ser una matrícula
        if int(cv2.contourArea(contour)) > area and cv2.contourArea(contour) < 8000:
            area = int(cv2.contourArea(contour))
            # Dibujar un rectángulo alrededor de la matrícula
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            for i in range(len(box)):
                if i == 0:
                    box[i][0] -= 12  # Disminuir x en 12 unidades
                    box[i][1] += 12  # Aumentar y en 12 unidades
                elif i == 1:
                    box[i][0] += 12  # Aumentar x en 12 unidades
                    box[i][1] += 12  # Aumentar y en 12 unidades
                elif i == 2:
                    box[i][0] += 12  # Aumentar x en 12 unidades
                    box[i][1] -= 12  # Disminuir y en 12 unidades
                elif i == 3:
                    box[i][0] -= 12  # Disminuir x en 12 unidades
                    box[i][1] -= 12  # Disminuir y en 12 unidades
            break

    # Draw the contours
    image_final = cv2.rectangle(
        image.copy(),
        tuple(box[0].astype(int)),
        tuple(box[2].astype(int)),
        (0, 0, 255),
        2,
    )
    show_image(image_final, "Final")

    # Crop the plate
    if box.size > 0:
        # Obtener las coordenadas mínimas (esquina superior izquierda)
        # y máximas (esquina inferior derecha)
        min_x, min_y = np.min(box, axis=0)
        max_x, max_y = np.max(box, axis=0)

        # Recortar la placa de la imagen original
        plate = image_final[min_y.astype(int): max_y.astype(
            int), min_x.astype(int): max_x.astype(int)]

    cv2.imwrite(os.path.join(os.path.dirname(__file__), "temp_plate.png"), plate)
    show_image(plate, "Final")

    return cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY), box
