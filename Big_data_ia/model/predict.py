from pathlib import Path

import numpy as np


MODEL_PATH = Path(__file__).resolve().parent / "saved_model" / "digit_model.keras"


def load_trained_model():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to load the trained model. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return tf.keras.models.load_model(MODEL_PATH)


def load_model():
    return load_trained_model()


def predict_digit(model, image):
    if image.shape == (28, 28):
        image = image.reshape(1, 28, 28, 1)

    elif image.shape == (28, 28, 1):
        image = image.reshape(1, 28, 28, 1)

    image = image.astype("float32")

    if image.max() > 1:
        image = image / 255.0

    prediction = model.predict(image, verbose=0)

    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    return predicted_digit, confidence, prediction[0]
