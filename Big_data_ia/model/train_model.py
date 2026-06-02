from pathlib import Path

from model.cnn_model import create_cnn_model
from utils.preprocessing import load_and_preprocess_mnist


MODEL_PATH = Path(__file__).resolve().parent / "saved_model" / "digit_model.keras"


def train_model():
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    model = create_cnn_model()

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_data=(x_test, y_test)
    )

    loss, accuracy = model.evaluate(x_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)

    print("Modèle sauvegardé :", MODEL_PATH)
    print(f"Précision finale : {accuracy * 100:.2f}%")

    return model, history


if __name__ == "__main__":
    train_model()
