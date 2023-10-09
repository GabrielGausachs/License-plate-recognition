import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os


config = {
    "print": {
        "original": False,
        "gray": False,
        "binary": False,
        "inverted": False,
        "character": False,
        "mask": False,
        "final": False,
    }
}


def show_image(image, title="Image"):
    if config["print"][title.lower()]:
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()


def erode(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iterations=iters)


def character_cleaner(img):
    # Apply threshold to get image with only black and white
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Invert image to get black letters on a white background
    inverted = cv2.bitwise_not(thresh)

    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    imageOut = img.copy()
    possible_contours = []

    for indx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Si el contorno no es el padre (el contorno exterior) y tiene un area mayor a 38
        if hierarchy[0][indx][3] != -1 and cv2.contourArea(cnt) > 40:
            possible_contours.append(cnt)

    possible_contours = sorted(
        possible_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

    for cnt in possible_contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(imageOut, [box], 0, (255, 0, 255), 1)

        (x, y, w, h) = cv2.boundingRect(cnt)
        letter = img[y: y + h, x: x + w]

    return letter


def segmentate_characters(input="temp_plate.png"):

    # Create temp folder if it doesn't exist
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "temp_digits")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "temp_digits"))
    else:
        # Delete all files in temp folder
        for file in os.listdir(os.path.join(os.path.dirname(__file__), "temp_digits")):
            os.remove(os.path.join(os.path.dirname(
                __file__), "temp_digits", file))

    image = cv2.imread(os.path.join(os.path.dirname(__file__), input))
    show_image(image, "Original")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=250)
    show_image(image, "Gray")

    threshold = 127
    detected = False
    while not detected:
        ret3, th3 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        show_image(th3, "binary")

        inverted = cv2.bitwise_not(th3)
        show_image(inverted, "Inverted")

        contours, hierarchy = cv2.findContours(
            inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        imageOut = image.copy()
        posible_contours = []

        characters = []  # List of characters

        for indx, cnt in enumerate(contours):
            if hierarchy[0][indx][3] == -1:
                x, y, w, h = cv2.boundingRect(cnt)

                if (
                    x != 0
                    and y != 0
                    and x + w != imageOut.shape[1]
                    and y + h != imageOut.shape[0]
                    and y > 23
                    and w/h > 0.15
                    and cv2.contourArea(cnt) > 45
                ):
                    posible_contours.append(cnt)

        posible_contours = sorted(
            posible_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])

        if len(posible_contours) > 6:
            detected = True
        n = 0
        margen = 5
        if detected:
            if len(posible_contours) > 7:  # E detected
                posible_contours = posible_contours[1:]
            else:
                pass
            for cnt in posible_contours:
                if n < 7:
                    imageOut = image.copy()
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.intp(box)
                    # cv2.drawContours(imageOut, [box], 0, (255, 0, 255), 2)

                    (x, y, w, h) = cv2.boundingRect(cnt)
                    x -= margen
                    y -= margen
                    w += 2 * margen
                    h += 2 * margen
                    # letter = image[y: y + h, x: x + w]
                    # letter = character_cleaner(letter)
                    # Para cada punto de la imagen original, si esta fuera del rectangulo, se pone en blanco
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    for i in range(image.shape[0]):
                        for j in range(image.shape[1]):
                            if not (x <= j <= x + w and y <= i <= y + h):
                                imageOut[i, j] = 255
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    x -= margen
                    y -= margen
                    w += 2 * margen
                    h += 2 * margen
                    letter = imageOut
                    # Threshold
                    ret, thresh = cv2.threshold(letter, 125, 255, cv2.THRESH_BINARY)
                    # Add some blurness
                    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
                    letter = thresh
                    letter = letter[y: y + h, x: x + w]
                    show_image(letter, "Mask")
                    characters.append(letter)
                    show_image(letter, "Character")
                    if letter is not None:
                        cv2.imwrite(
                            os.path.join(
                                os.path.dirname(__file__),
                                "temp_digits",
                                "temp_digit_" + str(n) + ".png",
                            ),
                            letter,
                        )
                        n += 1

            show_image(imageOut, "final")
            return characters
        else:
            threshold *= 0.9
            print("Threshold: ", threshold)
