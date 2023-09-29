import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os


config = {
    "print": {
        "original": True,
        "binary": True,
        "inverted": True,
        "character": True,
        "final": True,
    }
}


def show_image(image, title="Image"):
    if config["print"][title.lower()]:
        plt.imshow(image, cmap="gray")
        plt.title(title)
        plt.show()


def segmentate_characters(input):
    # Create temp folder if it doesn't exist
    if not os.path.exists("temp_digits"):
        os.makedirs("temp_digits")
    else:
        # Delete all files in temp folder
        for file in os.listdir("temp_digits"):
            os.remove(os.path.join("temp_digits", file))

    # image = cv2.imread(input)
    image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=250)
    show_image(image, "Original")

    ret3, th3 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    show_image(th3, "Binary")

    inverted = cv2.bitwise_not(th3)
    show_image(inverted, "Inverted")

    imageOut = image.copy()

    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    characters = []

    n = 0
    for indx, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        if (
            x != 0
            and y != 0
            and x + w != imageOut.shape[1]
            and y + h != imageOut.shape[0]
        ):
            if cv2.contourArea(cnt) > 140:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(imageOut, [box], 0, (255, 0, 255), 2)

                # Masking the part other than the number plate
                mask = np.zeros(imageOut.shape, np.uint8)
                imageOut = cv2.drawContours(
                    mask,
                    [box],
                    0,
                    255,
                    -1,
                )

                # Now crop
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                letter = imageOut[topx : bottomx + 1, topy : bottomy + 1]
                letter = imutils.resize(letter, width=25)

                characters.append(letter)

                show_image(letter, "Character")
                cv2.imwrite(str(n) + "-character.jpg", letter)
                n += 1

    show_image(imageOut, "Final")
    return characters


if __name__ == "__main__":
    segmentate_characters(cv2.imread("test.jpg"))
