import pandas as pd
import numpy as np
import requests

def client_code():
    """
    Function to make API requests and compare predictions with expected results.
    """
    # Make a GET request to the API with input parameters
    response = requests.get("http://127.0.0.1:5000/predictions_dict?url=https://www.ayaakov.co.il/files/products/product1596_image1_2022-01-03_11-07-20.jpg")
    # Extract the prediction value from the response
    prediction = response.text
    # Print the prediction and expected value
    print("Prediction:", prediction)


if __name__ == '__main__':
    client_code()
