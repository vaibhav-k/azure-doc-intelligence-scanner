from __future__ import annotations

import cv2
import numpy as np


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def reduce_noise(image: np.ndarray) -> np.ndarray:
    """Applies bilateral filtering to reduce noise while preserving sharp text edges."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Detects and corrects text skew angle in document images."""
    gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.nonzero(thresh > 0))
    if coords.size == 0:
        return image

    box = cv2.minAreaRect(coords)
    angle = float(box[2])
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    m = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def preprocess_image(file_path: str) -> bytes:
    """
    Loads an image from disk, applies deskewing, noise reduction, and contrast enhancement,
    and returns the processed image encoded as bytes ready for Azure Document Intelligence.

    Args:
        file_path: The path to the image file to be preprocessed.

    Raises:
        FileNotFoundError: If the image cannot be loaded from the specified path.
        ValueError: If the image cannot be encoded to bytes.

    Returns:
        bytes: The preprocessed image encoded as bytes.
    """
    image = cv2.imread(file_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at '{file_path}'.")

    deskewed = deskew_image(image)
    denoised = reduce_noise(deskewed)
    enhanced = enhance_contrast(denoised)

    success, encoded_image = cv2.imencode(".png", enhanced)
    if not success:
        raise ValueError("Failed to encode preprocessed image bytes.")

    return encoded_image.tobytes()
