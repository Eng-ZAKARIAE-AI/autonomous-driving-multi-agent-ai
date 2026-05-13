"""Camera processing utilities for CARLA observations."""

import cv2
import numpy as np
from typing import Optional, Tuple


def normalize_image(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


class CameraProcessor:
    def __init__(self, width: int = 84, height: int = 84):
        self.width = width
        self.height = height

    def process(self, image: Optional[np.ndarray]) -> np.ndarray:
        if image is None:
            return np.zeros((3, self.height, self.width), dtype=np.float32)

        resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)
        normalized = normalize_image(resized)
        transposed = normalized.transpose(2, 0, 1)
        return transposed.astype(np.float32)
