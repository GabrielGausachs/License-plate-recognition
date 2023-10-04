import Levenshtein


def test_helper(actual_characters, predicted_full, predicted_segmented):
    """
    Helper function to calculate accuracy and print results for a single test case.

    Args:
        actual_characters (str): The actual characters in the image.
        predicted_full (str): Predicted characters from full plate.
        predicted_segmented (str): Predicted characters from segmented plate.

    Returns:
        accuracy_full_plate (float): Accuracy for full plate.
        accuracy_segmented_plate (float): Accuracy for segmented plate.
    """
    # Calculate Levenshtein distances for full plate and segmented plate
    levenshtein_distance_full_plate = Levenshtein.distance(predicted_full, actual_characters)
    levenshtein_distance_segmented_plate = Levenshtein.distance(predicted_segmented, actual_characters)

    # Calculate accuracy for full plate
    accuracy_full_plate = sum(1 for a, b in zip(predicted_full, actual_characters) if a == b) / len(actual_characters) * 100

    # Calculate accuracy for segmented plate
    accuracy_segmented_plate = sum(1 for a, b in zip(predicted_segmented, actual_characters) if a == b) / len(actual_characters) * 100

    # Log results with Levenshtein distances
    print("\n--------------------")
    print(f'Actual characters: {actual_characters}')
    print("\nFull plate:")
    print(f"\nPredicted:               {predicted_full}")
    print(f"Levenshtein distance:    {levenshtein_distance_full_plate} - {'Correct' if accuracy_full_plate == 100 else 'Incorrect'}")
    print(f"Accuracy:                {accuracy_full_plate:.2f}% - {'Correct' if accuracy_full_plate == 100 else 'Incorrect'}")

    print("\nSegmented plate:")
    print(f"\nPredicted:               {predicted_segmented}")
    print(f"Levenshtein distance:    {levenshtein_distance_segmented_plate} - {'Correct' if accuracy_segmented_plate == 100 else 'Incorrect'}")
    print(f"Accuracy:                {accuracy_segmented_plate:.2f}% - {'Correct' if accuracy_segmented_plate == 100 else 'Incorrect'}")

    return accuracy_full_plate, accuracy_segmented_plate
