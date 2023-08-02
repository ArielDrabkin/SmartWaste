import predict
import gradio as gr
#from keras.models import load_model
import timeit
from PIL import Image
import numpy as np


def image_identity(image):
    image = Image.fromarray((image * 255).astype(np.uint8))

    # Define the class labels for the model's output
    output_class = ["Plastic bottle/Can to deposit in Supermarkets", "Big Cardboard bin", "Unrecyclable garbage",
                    "Glass - Purple bin", "Organic waste - Composter", "Grocery Packages - Orange bin",
                    "Paper - Blue bin"]

    # Measure the time taken to make predictions
    start_time = timeit.default_timer()

    # Make predictions and compare with expected results
    predicted_class1, predicted_prob1, predicted_class2, predicted_prob2 = predict.check_item(image)

    # Calculate and print the prediction time
    end_time = timeit.default_timer()
    pred_time = round(end_time - start_time, 2)
    prediction_time = f"{pred_time} seconds"
    prediction_1 = f"{predicted_class1}\n probability: {predicted_prob1:.2f}"
    prediction_2 = f"{predicted_class2}\n probability: {predicted_prob2:.2f}"
    return prediction_1, prediction_2, prediction_time


iface = gr.Interface(image_identity,
                     gr.inputs.Image(label="Upload your image", source="upload"),  # Adding a label to the input
                     [gr.outputs.Textbox(label="Prediction 1"),gr.outputs.Textbox(label="Prediction 2"), gr.outputs.Textbox(label="Prediction Time")],
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
    iface.launch(share=False)
