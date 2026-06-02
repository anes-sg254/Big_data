import numpy as np
import tensorflow as tf


MODEL_PATH = "model/saved_model/digit_model.keras"


def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_digit(model, image):
    if image.shape == (28, 28):
        image = image.reshape(1, 28, 28, 1)

    if image.shape == (28, 28, 1):
        image = image.reshape(1, 28, 28, 1)

    image = image.astype("float32")

    if image.max() > 1:
        image = image / 255.0

    prediction = model.predict(image, verbose=0)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return predicted_digit, confidence, prediction[0]