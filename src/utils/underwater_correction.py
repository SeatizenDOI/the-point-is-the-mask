# Code Adapt from https://github.com/nikolajbech/underwater-image-color-correction
import numpy as np
from PIL import Image


def get_color_filter_matrix(pixels, width, height):
    # Magic values
    num_of_pixels = width * height
    threshold_ratio = 2000
    threshold_level = num_of_pixels / threshold_ratio
    min_avg_red = 60
    max_hue_shift = 120
    blue_magic_value = 1.2
    blue_magic_value = 2.0
    blue_magic_value = 0

    # Compute average color
    avg = calculate_average_color(pixels, width, height)

    # Determine hue shift
    hue_shift = 0
    new_avg_red = avg[0]
    while new_avg_red < min_avg_red:
        shifted = hue_shift_red(avg, hue_shift)
        new_avg_red = sum(shifted)
        hue_shift += 1
        if hue_shift > max_hue_shift:
            new_avg_red = 60  # Max value

    # Create histogram
    hist = np.zeros((3, 256), dtype=int)  # Channels: R, G, B
    red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    shifted_red = hue_shift_red(np.stack([red, green, blue], axis=-1), hue_shift).sum(axis=-1)
    shifted_red = np.clip(shifted_red, 0, 255).astype(int)

    # Update histogram
    np.add.at(hist[0], shifted_red.flatten(), 1)
    np.add.at(hist[1], green.flatten(), 1)
    np.add.at(hist[2], blue.flatten(), 1)

    # Normalize histogram
    normalize = [np.array([0])] * 3  # Start with zero
    for i, channel_hist in enumerate(hist):
        normalize[i] = np.append(normalize[i], np.where(channel_hist < threshold_level + 2)[0])
        normalize[i] = np.append(normalize[i], 255)  # End with 255

    # Adjust histogram
    adjust = [normalizing_interval(norm) for norm in normalize]

    # Compute color gain and offset
    shifted = hue_shift_red(np.array([1, 1, 1]), hue_shift)
    gains = 256 / (np.array([adj["high"] - adj["low"] for adj in adjust]))
    offsets = (-np.array([adj["low"] for adj in adjust]) / 256) * gains

    # Apply blue magic value
    adjst_red = shifted[0] * gains[0]
    adjst_red_green = shifted[1] * gains[0]
    adjst_red_blue = shifted[2] * gains[0] * blue_magic_value

    return [
        adjst_red, adjst_red_green, adjst_red_blue, 0, offsets[0],
        0, gains[1], 0, 0, offsets[1],
        0, 0, gains[2], 0, offsets[2],
        0, 0, 0, 1, 0
    ]

def calculate_average_color(pixels, width, height):
    avg = np.array([
        pixels[:, :, 0].sum() / (width * height),
        pixels[:, :, 1].sum() / (width * height),
        pixels[:, :, 2].sum() / (width * height)
    ])
    return avg

def hue_shift_red(rgb, h):
    U = np.cos(h * np.pi / 180)
    W = np.sin(h * np.pi / 180)

    r = (0.299 + 0.701 * U + 0.168 * W) * rgb[..., 0]
    g = (0.587 - 0.587 * U + 0.330 * W) * rgb[..., 1]
    b = (0.114 - 0.114 * U - 0.497 * W) * rgb[..., 2]

    return np.stack([r, g, b], axis=-1)

def normalizing_interval(norm_array):
    diffs = np.diff(norm_array)
    max_dist_idx = np.argmax(diffs)
    return {"low": norm_array[max_dist_idx], "high": norm_array[max_dist_idx + 1]}

def apply_filter(data, filter):
    data = np.array(data, dtype=np.float32)

    # Apply filter transformations directly using slicing
    data[..., 0] = np.clip(data[..., 0] * filter[0] + data[..., 1] * filter[1] + data[..., 2] * filter[2] + filter[4] * 255, 0, 255)  # Red
    data[..., 1] = np.clip(data[..., 1] * filter[6] + filter[9] * 255, 0, 255)  # Green
    data[..., 2] = np.clip(data[..., 2] * filter[12] + filter[14] * 255, 0, 255)  # Blue

    return np.array(data, dtype=np.uint8)
def load_image_as_numpy(image_path):
    image = Image.open(image_path).convert("RGB")  # Convert to RGBA
    return np.array(image, dtype=np.uint8)

def save_image_from_numpy(pixels, output_path):
    # Ensure pixels is a valid RGBA image

    image = Image.fromarray(pixels, "RGB")  # Convert NumPy array to image
    image.save(output_path, format="JPEG")  # Save as PNG to keep RGBA channels
