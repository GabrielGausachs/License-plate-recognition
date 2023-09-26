# Signal, image and video processing

In this repository we will build diferents projects related to signal, images and video processing.

### License Plate Detection

In the License Plate Detection project, we aim to detect license plates in static images. The project involves implementing various techniques and algorithms to read digital images, detect regions of interest (ROIs), and finally detect the license plates' position in the images.

#### Dataset

For the License Plate Detection project, we used the Car License Plates dataset available on Kaggle. The dataset contains 433 images with around 1007 annotated license plates. We used a subset of 200 images with 430 annotated license plates for training and testing our model.

#### Approach

Our approach to detecting license plates involved the following steps:

1. Reading images and resizing them to reduce computational complexity
2. Conversion to grayscale
3. Edge detection with the Canny algorithm
4. Morphological operations to remove small edge fragments and join nearby edges
5. Contour detection
6. Filtering contours by area and aspect ratio
7. After filtering, resizing the contours
8. Using a Support Vector Machine (SVM) classifier to distinguish between true and false license plates

#### Results

Our approach was able to detect license plates with a 98% accuracy rate and an average processing time of 0.8 seconds per image.

#### Conclusion

The License Plate Detection project illustrates the effectiveness of combining various techniques and algorithms to detect ROIs in digital images. However, the SVM classifier's effectiveness depends on the quality of the training and testing data. Therefore, it is essential to have a comprehensive and well-annotated dataset when attempting to train an SVM classifier for license plate detection.
