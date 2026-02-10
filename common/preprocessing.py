"""
Shared DOPE image preprocessing for training and inference.
"""

import numpy as np
from PIL import Image
import torchvision.transforms as transforms


DOPE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
DOPE_NORMALIZE_STD = (0.229, 0.224, 0.225)

_TO_TENSOR = transforms.ToTensor()
_NORMALIZE_TO_TENSOR = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(DOPE_NORMALIZE_MEAN, DOPE_NORMALIZE_STD),
    ]
)


def ensure_rgb_uint8(image):
    """Convert image array to uint8 RGB, replicating grayscale channels if needed."""
    arr = np.asarray(image)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] >= 4:
        arr = arr[:, :, :3]

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def to_model_tensor(image):
    """Return normalized tensor expected by DOPE networks."""
    rgb = ensure_rgb_uint8(image)
    return _NORMALIZE_TO_TENSOR(Image.fromarray(rgb))


def to_rgb_tensor(image):
    """Return non-normalized RGB tensor for visualization/logging."""
    rgb = ensure_rgb_uint8(image)
    return _TO_TENSOR(rgb)
