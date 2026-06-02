import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import load_and_preprocess_mnist
from utils.noise import add_noise
from model.predict import load_trained_model, predict_digit


def test_multiple_noisy_images():
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()
    model = load_trained_model()

    indices = [0, 1, 2, 3, 4, 5]
    noise_factor = 0.35

    plt.figure(figsize=(10, 6))

    for i, index in enumerate(indices):
        original_image = x_test[index]
        noisy_image = add_noise(original_image, noise_factor=noise_factor)

        digit, confidence, probabilities = predict_digit(model, noisy_image)

        print(f"Image {index} → prédiction : {digit}, confiance : {confidence:.2f}%")

        plt.subplot(2, 3, i + 1)
        plt.imshow(noisy_image.reshape(28, 28), cmap="gray")
        plt.title(f"Prédit : {digit}\n{confidence:.2f}%")
        plt.axis("off")

    plt.suptitle("Tests sur plusieurs images bruitées")
    plt.show()


if __name__ == "__main__":
    test_multiple_noisy_images()