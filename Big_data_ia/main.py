import numpy as np
import matplotlib.pyplot as plt

from utils.preprocessing import load_and_preprocess_mnist
from utils.noise import add_noise
from model.predict import load_trained_model, predict_digit


def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()

    model = load_trained_model()

    image = x_test[0]
    noisy_image = add_noise(image, noise_factor=0.35)

    digit, confidence, probabilities = predict_digit(model, noisy_image)

    print("Chiffre prédit :", digit)
    print(f"Confiance : {confidence:.2f}%")
    print("Probabilités :", np.round(probabilities, 3))

    plt.imshow(noisy_image.reshape(28, 28), cmap="gray")
    plt.title(f"Prédiction : {digit} - Confiance : {confidence:.2f}%")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
