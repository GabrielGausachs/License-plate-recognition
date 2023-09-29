"""
Identify characters in a license plate
using Python Library Sklearn
"""

import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

import cv2
import os


def train_model():
    """
    Function to identify characters in a license plate
    
    Args:
        img: cv2 read image in B&W format
    Returns:
        result: string of characters
    """
    
    
    digits = "C:/Users/arnau/OneDrive/Escritorio/ARNAU G/UNI/4t ENG DADES\PSIV II/PSIV-projects-1/License plate detection/img/digits"
    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)

    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.5, shuffle=False
    )

    # Learn the digits on the train subset
    clf.fit(X_train, y_train)

    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")
        
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )
    
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")

    plt.show()
    
    # The ground truth and predicted lists
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix

    # For each cell in the confusion matrix, add the corresponding ground truths
    # and predictions to the lists
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
    
    
    
    return model
    

def test():
    """
    Function to test the identify_character function with the digits image files

    Args:
        img (string): image file path
    """
    total_correct = 0
    total = 0
    img_directory = '../img/digits/'
    file_directory = os.path.dirname(os.path.realpath(__file__))
    print(file_directory)
    for character in os.listdir(os.path.join(file_directory, img_directory)):
        print("Predicting character: ", character)
        for img in os.listdir(os.path.join(file_directory, img_directory + character)):
            img_path = os.path.join(file_directory, img_directory + character + '/' + img)
            img = cv2.imread(img_path, 0)
            predicted = identify_character(img)
            if predicted == character:
                total_correct += 1
            total += 1
    print('Accuracy: ', total_correct/total)
            
    
if __name__ == '__main__':
    train_model()