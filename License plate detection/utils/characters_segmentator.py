import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os


config = {
    "print": {
        "original": True,
        "gray": False,
        "binary": False,
        "inverted": False,
        "character": False,
        "final": True,
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

    ret3, th3 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    show_image(th3, "binary")

    inverted = cv2.bitwise_not(th3)
    show_image(inverted, "Inverted")


    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    imageOut = image.copy()
    posible_contours = []

    characters = []  # List of characters

    for indx, cnt in enumerate(contours):
        if hierarchy[0][indx][3] == -1 and hierarchy[0][indx][2] == -1:
            x, y, w, h = cv2.boundingRect(cnt)

            if (
                x != 0
                and y != 0
                and x + w != imageOut.shape[1]
                and y + h != imageOut.shape[0]
                and cv2.contourArea(cnt) > 38
            ):
                # if w > 25:
                #     # Calculate the center of the bounding rectangle
                #     center_x = x + w // 2
                #     center_y = y + h // 2
                    
                #     # Create two new contours by splitting the original contour
                #     contour1 = cnt[:, :center_x, :]
                #     contour2 = cnt[:, center_x:, :]
                #     posible_contours.append(contour1)
                #     posible_contours.append(contour2)

                # else:
                    posible_contours.append(cnt)

    posible_contours = sorted(
        posible_contours, key=lambda cnt: cv2.boundingRect(cnt)[0])
    n = 0
    margen = 5

    if len(posible_contours) > 7:  # E detected
        posible_contours = posible_contours[1:]
    else:
        pass
    for cnt in posible_contours:
        if n < 7:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            cv2.drawContours(imageOut, [box], 0, (255, 0, 255), 2)

            (x, y, w, h) = cv2.boundingRect(cnt)
            print(w)
            x -= margen
            y -= margen
            w += 2 * margen
            h += 2 * margen
            letter = image[y: y + h, x: x + w]
            characters.append(letter)
            show_image(letter, "Character")
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
