

# provar de fer la dilatacio abans que la binarització
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os

config = {
    "print": {
        "original": True,
        "gray": False,
        "blackhat": True,
        "step": False,
        "binary": False,
        "final": True,
        "edges": True,
    }
}


def show_image(image, title="Image"):
    if config["print"][title.lower()]:
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        plt.title(title)
        plt.show()
        
        
def dilate(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.dilate(img, kernel, iters)


def erode(img, kernel_size, iters):
    kernel = np.ones(kernel_size, np.uint8)
    return cv2.erode(img, kernel, iters)


def process_and_detect(img):
    height, width, _ = img.shape
    kernel = np.ones((2, 2), np.uint8)
    
    # Transform to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # print max and min values
    print("Max: ", np.max(img))
    print("Min: ", np.min(img))

    # Threshold
    _, mask = cv2.threshold(img, thresh=200, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_mask = cv2.bitwise_and(img, mask)

    # Edge Detection
    edges = cv2.Canny(img_mask, height, width)
    show_image(edges, "Edges")

    # Contours
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    NumberPlateCnt = None
    found = False
    lt, rb = [10000, 10000], [0, 0]

    # Calculate polygonal curve, see if it has 4 curves
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 4:
            found = True
            NumberPlateCnt = approx
            break

    if found:
        for point in NumberPlateCnt:
            cur_cx, cur_cy = point[0][0], point[0][1]
            if cur_cx < lt[0]: lt[0] = cur_cx
            if cur_cx > rb[0]: rb[0] = cur_cx
            if cur_cy < lt[1]: lt[1] = cur_cy
            if cur_cy > rb[1]: rb[1] = cur_cy

        crop = img_mask[lt[1]:rb[1], lt[0]:rb[0]]
        crop_res = img[lt[1]:rb[1], lt[0]:rb[0]]
    else:
        crop = img_mask.copy()
        crop_res = img.copy()

    return crop, crop_res, edges, NumberPlateCnt

# Uso de la función process_and_detect en un bucle
carpeta_imagenes = "../img/plates"

i = 0

for nombre_archivo in os.listdir(os.path.join(os.path.dirname(os.path.realpath(__file__)), carpeta_imagenes)):
    if i == 1:
        break
    ruta_completa = os.path.join(os.path.dirname(os.path.realpath(__file__)), carpeta_imagenes, nombre_archivo)
    print(ruta_completa)
    image = cv2.imread(ruta_completa)
    show_image(image, "Original")
    image = imutils.resize(image, width=1000)

    crop, crop_res, edges, screenCnt = process_and_detect(image)

    # Dibuja los contornos en la imagen original
    if screenCnt is not None:
        image_final = cv2.drawContours(image.copy(), [screenCnt], -1, (0, 255, 0), 3)
        show_image(image_final, "Final")

    i += 1