import pandas as pd
import pytesseract
import cv2
import time

#-----------------USING PYTESSERACT DIRECTLY WITH THE LICENSE PLATE-------------------

pytesseract.pytesseract.tesseract_cmd=r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

image = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/PSIV-projects/License plate detection/final_code/img/matricula.jpg')

# Run tesseract OCR on image
text = pytesseract.image_to_string(image)

# Print recognized text
print(text)


#----------------ANOTHER APPROACH----------------


