from keras.models import load_model
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import timeit
# Define the class labels for the model's output
output_class = ["Plastic bottle/Can to deposit in Supermarkets", "Big Cardboard bin", "Unrecyclable garbage", "Glass - Purple bin", "Organic waste - Composter", "Grocery Packages - Orange bin", "Paper - Blue bin"]

# Load the pre-trained model
model = load_model("vgg16_model_after_one_epoch.h5")


def predict(file, is_url=False):
    """
    Function to predict the class labels and probabilities of an input image.

    Args:
        file (str): Path to the input image file or URL.
        is_url (bool): Indicates whether the file argument is a URL or local file path.

    Returns:
        dict: A dictionary containing the predicted class labels as keys and
              their corresponding probabilities as values.
    """
    if is_url is True:
        # Download the image from the URL
        response = requests.get(file)
        img = Image.open(BytesIO(response.content))
    else:
        # Open the image file from the local file path
        img = Image.open(file)

    # Resize the image to match the input size expected by the model (224x224)
    resized_image = img.resize((224, 224))

    # Normalize the pixel values to be between 0 and 1
    normalized_image = np.array(resized_image) / 255.0

    # Expand dimensions to match the input shape expected by the model
    pred_image = np.expand_dims(normalized_image, axis=0)

    # Make predictions using the model
    predicted_array = model.predict(pred_image)

    # Create a dictionary of predicted class labels and probabilities
    pred_labels_and_probs = {
        output_class[i]: float(predicted_array[0][i])
        for i in range(len(output_class))
    }

    return pred_labels_and_probs


def check_item(file_path, is_url=False):
    """
    Function to make predictions on an input image and compare the results with the expected classes.

    Args:
        file_path (str): Path to the input image file or URL.
        is_url (bool): Indicates whether the file argument is a URL or local file path.
    """
    # Obtain the predicted class labels and probabilities
    prediction_dict = predict(file_path, is_url)

    # Get the index of the highest prediction
    ind = np.argmax(list(prediction_dict.values()))
    predicted_class = list(prediction_dict.keys())[ind]
    predicted_prob = list(prediction_dict.values())[ind]
    print(f"1st Prediction probabilities: {predicted_class} {predicted_prob:.2f}")

    # Get the index of the second highest prediction
    sorted_probs = np.sort(list(prediction_dict.values()))
    ind2 = np.argsort(list(prediction_dict.values()))[-2]
    predicted_class2 = list(prediction_dict.keys())[ind2]
    predicted_prob2 = sorted_probs[-2]
    print(f"2nd Prediction probabilities: {predicted_class2} {predicted_prob2:.2f}")


if __name__ == "__main__":
    # Prompt the user to enter the file path to the input image
    file_path = input("Enter the file path or URL to the input image: ")

    # Prompt the user to indicate if the input is a URL or a local file path
    is_url = input("Is the input a URL? (y/n): ").lower() == "y"

    # Measure the time taken to make predictions
    start_time = timeit.default_timer()

    # Make predictions and compare with expected results
    check_item(file_path, is_url)

    # Calculate and print the prediction time
    end_time = timeit.default_timer()
    pred_time = round(end_time - start_time, 2)
    print("Prediction time:", pred_time, "seconds")
