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
import copy
import time

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

#print(train_dataset.samples[0])
#print(len(train_dataset))
#print(len(train_dataset.classes))

batch_size=64

dataloaders_dict={'train':torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                  'val': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)}


import numpy as np
from PIL import Image

def show_images(dataset):
    # show some images
    plt.figure(figsize=(16, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        idx = np.random.randint(0,len(dataset.samples))
        image_path = dataset.samples[idx][0]
        label = os.path.basename(os.path.dirname(image_path))
        try:
            image = Image.open(image_path)
            plt.imshow(image)
            plt.title(label)
            plt.axis('off')
        except Exception as e:
            print(f"Error opening image at {image_path}: {e}")
    plt.show()

#show_images(train_dataset)


import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch.nn as nn
from torchvision import datasets, models, transforms

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model_fe(num_classes):
    # Resnet18 with pretrained weights 
    model = models.resnet18(pretrained=True) # Notice we are now loading the weights of a ResNet model trained on ImageNet
    set_parameter_requires_grad(model,True)
    model.fc = nn.Linear(in_features=512,out_features=num_classes)  
    input_size = 224
    return model, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    acc_history = {"train": [], "val": []}
    losses = {"train": [], "val": []}

    # we will keep a copy of the best weights so far according to validation accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    losses[phase].append(loss.item())

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            acc_history[phase].append(epoch_acc.item())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, acc_history, losses

num_classes = len(train_dataset.classes)
model, input_size = initialize_model_fe(num_classes)

params_to_update = []
for name,param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)

# Send the model to GPU
model = model.to(device)
# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

train_model(model,dataloaders_dict,criterion,optimizer_ft,num_epochs)