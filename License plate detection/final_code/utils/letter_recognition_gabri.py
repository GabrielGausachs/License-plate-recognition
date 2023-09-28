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


#----------------WITH EASYOCR----------------
"""
import easyocr

reader = easyocr.Reader(["es"])
image = cv2.imread('C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/PSIV-projects/License plate detection/final_code/img/1letter.jpg')
result = reader.readtext(image)
print(result)

"""


#----------------LINEARSVC-------------------

from sklearn.svm import SVC, LinearSVC
import os
import shutil
import random
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

path = 'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/CNN letter Dataset'
def load_images(path):

    # Directorio donde se guardarán los conjuntos de entrenamiento, validación y prueba
    directorio_destino = 'C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/splits'

    # Crear directorios para entrenamiento, validación y prueba
    os.makedirs(os.path.join(directorio_destino, 'train'), exist_ok=True)
    os.makedirs(os.path.join(directorio_destino, 'validation'), exist_ok=True)
    os.makedirs(os.path.join(directorio_destino, 'test'), exist_ok=True)

    # Porcentaje de datos para cada conjunto
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10

    # Recorre todas las carpetas en el directorio principal
    for carpeta_letra_numero in os.listdir(path):
        carpeta_actual = os.path.join(path, carpeta_letra_numero)
        
        if os.path.isdir(carpeta_actual):
            archivos = os.listdir(carpeta_actual)
            random.shuffle(archivos)
            
            total_archivos = len(archivos)
            train_split = int(total_archivos * train_ratio)
            validation_split = int(total_archivos * validation_ratio)
            
            train_files = archivos[:train_split]
            validation_files = archivos[train_split:train_split + validation_split]
            test_files = archivos[train_split + validation_split:]
            
            # Copiar archivos a los directorios correspondientes
            for archivo in train_files:
                origen = os.path.join(carpeta_actual, archivo)
                destino = os.path.join(directorio_destino, 'train', carpeta_letra_numero)
                os.makedirs(destino, exist_ok=True)
                shutil.copy(origen, destino)
            
            for archivo in validation_files:
                origen = os.path.join(carpeta_actual, archivo)
                destino = os.path.join(directorio_destino, 'validation', carpeta_letra_numero)
                os.makedirs(destino, exist_ok=True)
                shutil.copy(origen, destino)
            
            for archivo in test_files:
                origen = os.path.join(carpeta_actual, archivo)
                destino = os.path.join(directorio_destino, 'test', carpeta_letra_numero)
                os.makedirs(destino, exist_ok=True)
                shutil.copy(origen, destino)

    print("Proceso completado. Los conjuntos de datos se han creado en", directorio_destino)


#load_images(path)

#resize =(75,50) perque es el aprox size dels digits segmentats
train_transforms =transforms.Compose([
    transforms.Resize((75,50)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transforms =transforms.Compose([
    transforms.Resize((75,50)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_path = "C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/splits/train"
valid_path = "C:/Users/Gabriel/OneDrive/Escritorio/4t any uni/psiv/splits/validation"


train_dataset = torchvision.datasets.ImageFolder(root = train_path, transform = train_transforms )
valid_dataset = torchvision.datasets.ImageFolder(root = valid_path, transform = valid_transforms )

print(train_dataset)

batch_size=64

dataloaders_dict={'train':torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4),
                  'val': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)}


import numpy as np
from PIL import Image

# show some images
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    idx = np.random.randint(0,len(train_dataset.samples))
    image_path = train_dataset.samples[idx][0]
    try:
        image = Image.open(image_path)
        plt.imshow(image)
        plt.axis('off')
    except Exception as e:
        print(f"Error opening image at {image_path}: {e}")

plt.show()