import numpy as np


def add_noise(image, noise_factor=0.3):
    noisy_image = image + noise_factor * np.random.normal(
        loc=0.0,
        scale=1.0,
        size=image.shape
    )

    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    return noisy_image