import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import os

carpeta_imagenes = "C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/car images"
foto = "C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/car images/PXL_20210921_094938026.jpg"


def character_segmentation(input):

    image = cv2.imread(input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=250)
    cv2.imshow('box', image)
    cv2.waitKey(0)

    ret3,th3 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    cv2.imshow('box',th3)
    cv2.waitKey(0)
    inverted = cv2.bitwise_not(th3)
    cv2.imshow('box',inverted)
    cv2.waitKey(0)

    imageOut=image.copy()

    contours,hierarchy = cv2.findContours(inverted,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #print(imageOut.shape)
    characters=[]
    n=0
    for indx, cnt in enumerate(contours):
        x,y,w,h=cv2.boundingRect(cnt)

        if x!=0 and y!=0 and x+w!=imageOut.shape[1] and y+h!=imageOut.shape[0]:
            #print(cv2.contourArea(cnt))
            if cv2.contourArea(cnt)>140:
                rect=cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                cv2.drawContours(imageOut,[box],0,(255,0,255),2)

                # Masking the part other than the number plate
                mask = np.zeros(imageOut.shape,np.uint8)
                new_image = cv2.drawContours(mask,[box],0,255,-1,)
                new_image = cv2.bitwise_and(image,image,mask=mask)

                # Now crop
                (x, y) = np.where(mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                letter = imageOut[topx:bottomx+1, topy:bottomy+1]
                letter = imutils.resize(letter, width=25)

                cv2.imshow('box', letter)
                cv2.imwrite(str(n)+'letter.jpg', letter)
                cv2.waitKey(0)
                n+=1

    cv2.imshow('rect',imageOut)
    cv2.waitKey(0)

character_segmentation('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/PSIV-projects/License plate detection/final_code/img/matricula.jpg')