# Signal, image and video processing

In this repository we will build diferents projects related to signal, images and video processing.

### License Plate Detection

This project is dedicated to the task of detecting license plates in static images. License plate detection plays a crucial role in various applications, including traffic management, law enforcement, and automated toll collection. The project involves implementing various techniques and algorithms to read digital images, detect regions of interest and detect the license plates' position in the images to finally extract the license number of the car.

#### Dataset

For the License Plate Detection project, we used the Car License Plates dataset available on Kaggle. The dataset contains 433 images with around 1007 annotated license plates. We used a subset of 200 images with 430 annotated license plates for training and testing our model. Also, we used a dataset of 17 images taken from our University parking. 

For training our models, we used a big dataset with a big number of photos of each character that can be in a license plate, from '0' to 'Z'.

#### Step 1: Detecting License Plate from an image
In the first step of the project, we aim to detect the license plate from our images using mostly Opencv. Here is the process we followed:
1. Convert Image to Grayscale using **cvtColor** function
2. **Erosion** in two directions, first vertically and then horizontally. This process helps in enhancing the edges of the characters on the license plate.
3. A morphological operation called **Blackhat** is applied to the eroded image. This operation highlights the dark regions against a light background, potentially emphasizing the characters on the license plate.
4. A **dilation** operation is applied to the blackhat result. This operation expands the highlighted regions, making them more prominent.
5. A binary **thresholding** operation is performed on the dilated image. This step creates a binary image with distinct foreground (character regions) and background.
6. Contours are detected in the binary image using OpenCV's **findContours** function. The largest 10 contours (based on area) are selected for further analysis.
7. Loop through the selected contours and get the biggest area.
8. A red bounding box is drawn around the potential license plate on a copy of the original image.
9. If a potential license plate is found, its coordinates are used to crop the license plate region from the original image and we save it as 'temp_plate.jpg'.

This process can be found in the file called plate_dectector.py

![Alt text](<License plate detection/img/plates/0182GLK.png>)

![Alt text](<License plate detection/img/figure1.png>)

#### Step 2: Character segmentation
After getting the original image cropped, it's time to detect the characters in the license plate. To achieve this, we follow this process:
1. A binary **thresholding** operation is performed on the cropped image, previously converted it to Grayscale.
2. The binary image is inverted  using a **bitwise NOT** operation to ensure characters are represented as white regions on a black background.
3. We use **findContours** to find the differents contours in the image. We use hierarchy to get from the rectangular contour that is the license plate the inside contours that represent the characters.
4. We filter this contours so that we get the ones that are not in one end and have a bigger area than an especific number.
5. After that, if the number of contours is bigger than 7, it means that the *E* from Spain in the license plate is detected, so we sort this list of contours by the x value, so that the first contour is the leftmost contour in the image and we removed this contour. If it is not bigger than 7, it means that we achieved previously the removal of the *E*.
6. We loop throught the detected contours and we fit a rotated box around them and crop them adding a margins so that we get the character perfectly.
7. We return a list of images where each image is one character.

This process can be found in the file called character_segmentator.py

![Alt text](<License plate detection/img/figure2.png>)


#### Step 3: Predicting the license plate
This is the last part of the process. Here, using Deep Learning models trained by us and defined OCR we try to predict the character from the image.

The first approach is using the **Tesseract** library. This library has a function called **image_to_string** that receive an image and the function detect embedded characters and return them as string.

For our models, we used for training the dataset of the characters mentioned above. We split the dataset in train and validation. Our **first model** is a sequential neural network using Keras that has a series of dense layers with ReLU activations and batch normalization. The final layer uses softmax activation to produce class probabilities for the 35 possible output classes. The **second model** is similar to the model before but we add data augmentations, rescaling and max-pooling. For the **last model**, we use transfer learning. We initialize a pretrained Resnet-18 model for feature extraction, where the pretrained weights are not updated but we add a final classification layer to adapt it to our problem.

#### Results

Our approach was able to detect license plates with a 98% accuracy rate and an average processing time of 0.8 seconds per image.

FALTA A FER

#### Conclusion

Our license plate detection project leverages the power of OpenCV and deep learning in Python to accurately identify license plates in static images. By combining computer vision techniques with convolutional neural networks, we've developed a robust and versatile solution for various applications, such as traffic management and law enforcement. This project showcases the potential of advanced technologies in automating the crucial task of license plate recognition, enhancing efficiency and accuracy in real-world scenarios.
