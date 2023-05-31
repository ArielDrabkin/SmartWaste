import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_vgg16_model():
    vgg16 = VGG16(input_shape=(224, 224, 3), weights="imagenet", include_top=False)

    # Freeze all layers in base model
    for layer in vgg16.layers:
        layer.trainable = False

    # Change classifier head with random seed for reproducibility
    x = layers.Flatten()(vgg16.output)
    x = layers.Dense(units=512, activation="relu")(x)
    prediction = layers.Dense(units=6, activation="softmax")(x)
    model = tf.keras.models.Model(inputs=vgg16.input, outputs=prediction)

    return model
