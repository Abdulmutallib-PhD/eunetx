# utils/yellow_mask_converter.py

from PIL import Image
import numpy as np
import os


def convert_yellow_to_mask(input_path, output_path):
    """
    Converts an annotated image with yellow pen (RGB [255, 255, 0])
    into a binary mask and saves as PNG.
    """
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)

    # Detect yellow pixels
    yellow_mask = (data[:, :, 0] == 255) & (data[:, :, 1] == 255) & (data[:, :, 2] == 0)

    # Convert to binary mask (255 for foreground, 0 for background)
    binary_mask = yellow_mask.astype(np.uint8) * 255

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(binary_mask).save(output_path)
    print(f"Mask saved to {output_path}")
