def load_and_preprocess_mnist():
    try:
        import tensorflow as tf
        from tensorflow.keras.utils import to_categorical
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to load the MNIST dataset. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test
