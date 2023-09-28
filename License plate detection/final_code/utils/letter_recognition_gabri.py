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

class ConvNet(nn.Module):
    def __init__(self,num_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2)
        self.bn6 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)
        self.conv7 = nn.Conv2d(64, 128, kernel_size=4, padding=0)
        self.bn7 = nn.BatchNorm2d(128)
        self.fc = nn.Linear(11648, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.dropout1(x)
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.relu(self.bn5(self.conv5(x)))
        x = self.relu(self.bn6(self.conv6(x)))
        x = self.dropout2(x)
        x = self.relu(self.bn7(self.conv7(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = self.softmax(x)
        return x

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
model = ConvNet(num_classes)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Number of epochs to train for 
num_epochs = 15

optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

train_model(model,dataloaders_dict,criterion,optimizer_ft,num_epochs)