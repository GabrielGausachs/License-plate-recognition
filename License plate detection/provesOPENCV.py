import cv2
import numpy as np
#import pytesseract

placa = []

image = cv2.imread('C:/Users/arnau/OneDrive/Escritorio/ARNAU G/UNI/4t ENG DADES/PSIV II/Lateral/PXL_20210921_094926329.jpg')
#image = cv2.imread('License plate detection/img2.jpg')
image = cv2.resize(image,(800,600))
#image = image[250:500,100:500]
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.blur(gray,(3,3))
canny = cv2.Canny(gray,30,200)
canny = cv2.dilate(canny,None,iterations=1)

cv2.imshow("car",canny)
cv2.waitKey(0)

cnts,_ = cv2.findContours(canny,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image,cnts,-1,(0,255,0),2)

cv2.imshow("car",image)
cv2.waitKey(0)

area_ct = 0

for c in cnts:
    area = cv2.contourArea(c)

        
    x,y,w,h = cv2.boundingRect(c)
    epsilon = 0.019*cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,epsilon,True)

    if len(approx)==4 and area>area_ct:  
        area_ct = area
        contorn = c
        #print('area=',area)
        
mask = np.zeros(gray.shape, np.uint8)
new_image = cv2.drawContours(mask, [contorn], 0,255, -1)
new_image = cv2.bitwise_and(gray, gray, mask=mask)
cv2.imshow('Imagen con Contornos', new_image)
cv2.waitKey(0)

