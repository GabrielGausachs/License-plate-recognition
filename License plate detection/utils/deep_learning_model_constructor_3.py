

#----------------RESNET-------------------
import pandas as pd
import time
import os
import torch 
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import copy
import time
import splitfolders
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, models, transforms

def load_images(path):

    file_directory = os.path.dirname(os.path.realpath(__file__))

    path = "output"

    output_path= os.path.join(file_directory, path)

    # Split folders
    splitfolders.ratio(path, output=output_path, seed=1337, ratio=(.8, 0.2))

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

    train_dataset = torchvision.datasets.ImageFolder(root = "./output/train/", transform = train_transforms )
    valid_dataset = torchvision.datasets.ImageFolder(root = "./output/val/", transform = valid_transforms )

    batch_size=64

    dataloaders_dict={'train':torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                  'val': torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)}

    return dataloaders_dict

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


def execute_model(full_path):
    dataloaders_dict = load_images(full_path)

    num_classes = 35

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
    num_epochs = 7

    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

    model, acc_history, losses = train_model(model,dataloaders_dict,criterion,optimizer_ft,num_epochs)

    torch.save(model.state_dict(),'resnet_gabri.pth')

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    path = '../img/digits'

    file_directory = os.path.dirname(os.path.realpath(__file__))

    full_path = os.path.join(file_directory, path)

    execute_model(full_path)