import pickle
import pandas as pd
import numpy as np
from flask import Flask, request
import gradio as gr
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import numpy as np
from io import BytesIO
from timeit import default_timer as timer
from model import create_vgg16_model
import json

output_class = ["cardboard", "glass", "trash", "metal", "paper", "plastic"]

vgg16_model = create_vgg16_model()


def predict(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))

    resized_image = img.resize((224, 224))
    normalized_image = np.array(resized_image) / 255.0
    pred_image = np.expand_dims(normalized_image, axis=0)

    predicted_array = vgg16_model.predict(pred_image)

    pred_labels_and_probs = {output_class[i]: float(predicted_array[0][i]) for i in range(len(output_class))}

    return pred_labels_and_probs


def classification(output_class, url, straps):
    predictions_dict = {key: [] for key in output_class}
    for i in range(straps):
        pred = predict(url)
        for trash in output_class:
            predictions_dict[trash].append(pred[trash])
    waste_class = {}
    for key, values in predictions_dict.items():
        mean = sum(values) / len(values)
        print(f"Mean for {key}: {mean:.3f}")
        waste_class[key] = round(mean, 3)
    return waste_class


app = Flask(__name__)


@app.route('/predictions_dict', methods=['GET'])
def predict_waste():
    """
    Predict the waste class.
    """
    url = request.args.get('url')
    waste_class = classification(output_class, url, 15)
    json_smartwaste = json.dumps(waste_class)
    return json_smartwaste


if __name__ == '__main__':
    app.run(debug=True)
