import predict
import gradio as gr
import timeit
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import json


def image_identity(image):
    """
    Take an image as input and return prediction information and an image of the predicted waste bin.

    Parameters:
    image (np.array): An image in numpy array format.

    Returns:
    bin_image1 (np.array): An image of the predicted waste bin.
    predicted_class1 (str): The class of the predicted bin.
    predicted_prob1 (float): The probability of the predicted class.
    predicted_class2 (str): The second most likely class.
    predicted_prob2 (float): The probability of the second most likely class.
    prediction_time (str): The time taken to make the prediction.
    """
    image = Image.fromarray((image * 255).astype(np.uint8))

    # Load the classification and image data from the JSON file
    with open('Bins.json', 'r') as f:
        bins_dict = json.load(f)

    # Retrieve the class labels and bin images from the dictionary
    output_class = bins_dict["output_class"]
    bin_images = bins_dict["bin_images"]

    # Record the start time before making predictions
    start_time = timeit.default_timer()

    # Make predictions on the provided image
    predicted_class1, predicted_prob1, predicted_class2, predicted_prob2 = predict.check_item(image)

    # Record the end time after making predictions and calculate the total time taken
    end_time = timeit.default_timer()
    pred_time = round(end_time - start_time, 2)
    prediction_time = f"{pred_time} seconds"

    # Retrieve the image URL of the predicted bin
    bin_1 = output_class.index(predicted_class1)
    bin_image_url1 = bin_images[bin_1]
    response1 = requests.get(bin_image_url1)

    # Open the image and convert to a numpy array for returning
    bin_image1 = Image.open(BytesIO(response1.content))
    bin_image1 = np.array(bin_image1)

    return bin_image1, predicted_class1, predicted_prob1, predicted_class2, predicted_prob2, prediction_time


iface = gr.Interface(image_identity,
                     gr.inputs.Image(label="Upload your image", source="upload"),  # Adding a label to the input
                     [gr.outputs.Image(type="numpy", label="Predicted bin"), gr.outputs.Textbox(label="Prediction 1"),
                      gr.outputs.Textbox(label="Probability"),
                      gr.outputs.Textbox(label="Prediction 2"), gr.outputs.Textbox(label="Probability"),
                      gr.outputs.Textbox(label="Prediction Time")],
                     # Adding labels to the outputs
                     title="SmartWaste",  # Adding a title
                     # Adding a description
                     description="This application is an image recognition app that identifies recyclable objects and provides feedback on the appropriate bin for waste disposal.",
                     theme="Ajaxon6255/Emerald_Isle",  # Changing the theme
                     examples=[['images/bottle2.jpg'],
                               ['images/can2.jpg'],
                               ['images/wine2.jpg'],
                               ['images/compost2.jpg']],
                     cache_examples=False)  # Disabling caching

if __name__ == "__main__":
    # Launch the interface
    iface.launch(share=False)
