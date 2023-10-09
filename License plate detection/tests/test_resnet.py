
import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.deep_learning_model_constructor_3 import initialize_model_fe
from utils.plate_detector import find_plate
from utils.characters_segmentator import segmentate_characters
from test_helper import check_error

def show_image(image, title="Image"):
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.show()

def validate_model_resnet(validate_with_full_plate=False, print_compare_img=False, model=None):
    """
    Function to test the identify_character function with the digits image files

    Args:
        img (string): image file path
    """

    total_img = 0
    total_letters = 0
    total_correct_full_plate = 0
    total_correct_segmented_plate = 0

    img_directory = "../img/plates/"
    file_directory = os.path.dirname(os.path.realpath(__file__))

    predicted_full = None

    for test_img in os.listdir(os.path.join(file_directory, img_directory)):

        print("\n--------------------")
        print(f"Test image: {test_img}")

        actual_characters = test_img.split(".")[0]
        img_path = os.path.join(file_directory, img_directory + test_img)

        # Find plate
        plate, _ = find_plate(img_path)

        # Segmentate characters
        segmentated_chars = segmentate_characters("temp_plate.png")

        # Predict characters in a segmented plate
        predicted_segmentated = []
        loaded_model.eval()
        for i, character in enumerate(segmentated_chars):
            try:
                character_result = loaded_model(character)
                print('character',character_result)
            except Exception:
                print('Error')
            predicted_segmentated.append(''.join(e for e in character_result))

        predicted_segmentated = "".join(predicted_segmentated).replace(" ", "")

        # Call test_helper for accuracy calculation and logging
        accuracy_full_plate, accuracy_segmented_plate = check_error(actual_characters, predicted_full, predicted_segmentated)

        if print_compare_img:
            show_image(plate, "Final")

        # Check if correct for full plate
        if accuracy_full_plate == 100:
            total_correct_full_plate += 1

        # Check if correct for segmented plate
        if accuracy_segmented_plate == 100:
            total_correct_segmented_plate += 1

        total_letters += len(actual_characters)
        total_img += 1

    # Calculate overall accuracy percentages
    overall_percentage_accuracy_full_plate = (total_correct_full_plate / total_img) * 100
    overall_percentage_accuracy_segmented_plate = (total_correct_segmented_plate / total_img) * 100

    # Print overall test results
    print("\n--------------------")
    print(f"Total correct for full plate: {total_correct_full_plate} - {overall_percentage_accuracy_full_plate:.2f}%")
    print(f"Total correct for segmented plate: {total_correct_segmented_plate} - {overall_percentage_accuracy_segmented_plate:.2f}%")

    # Check overall test result based on overall accuracy percentages
    if overall_percentage_accuracy_full_plate >= 90 and overall_percentage_accuracy_segmented_plate >= 90:
        print("\n------------------ Test passed! ------------------")
        return True
    else:
        print("\n------------------ Test failed! ------------------")
        return False
    
if __name__ == "__main__":

    num_classes = 35
    loaded_model, input_size = initialize_model_fe(num_classes)

    loaded_model.load_state_dict(torch.load('../utils/saved_models/resnet_gabri.pth'))

    print('hello')

    validate_model_resnet(False,True, loaded_model)

    
