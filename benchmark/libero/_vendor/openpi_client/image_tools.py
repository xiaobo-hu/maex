"""Image preprocessing adapted from OpenPI's Apache-2.0 client."""

from __future__ import annotations

import numpy as np
from PIL import Image


def convert_to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert floating-point image data to compact uint8 values."""
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    return image


def resize_with_pad(
    images: np.ndarray,
    height: int,
    width: int,
    method=Image.BILINEAR,
) -> np.ndarray:
    """Resize images without distortion and zero-pad to the target size."""
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(image), height, width, method) for image in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad_pil(
    image: Image.Image,
    height: int,
    width: int,
    method: int,
) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = image.resize((resized_width, resized_height), resample=method)
    output = Image.new(resized.mode, (width, height), 0)
    output.paste(
        resized,
        (max(0, (width - resized_width) // 2), max(0, (height - resized_height) // 2)),
    )
    return output
