"""
Applying covariance to training images
"""
import argparse
import math
import os
import random

import albumentations as A
import cv2
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--ti_folder", type=str, default="TI/", help="training image to be augmented folder path")
parser.add_argument("--output_folder", type=str, default="generative_models/augmented/", help="output folder path")
parser.add_argument("--img_size", type=int, default=150, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

os.makedirs(opt.output_folder, exist_ok=True)

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.RandomRotate90(p=0.4),
    A.GaussianBlur(p=0.6),
    A.ShiftScaleRotate(p=0.5),
    A.GaussNoise(p=0.5),
    A.Cutout(num_holes=10, max_h_size=16, max_w_size=16)
])


def resize_linear(image_matrix, new_height: int, new_width: int):
    """Perform a pure-numpy linear-resampled resize of an image."""
    output_image = np.zeros((new_height, new_width), dtype=image_matrix.dtype)
    original_height, original_width = image_matrix.shape
    inv_scale_factor_y = original_height / new_height
    inv_scale_factor_x = original_width / new_width

    # This is an ugly serial operation.
    for new_y in range(new_height):
        for new_x in range(new_width):
            # If you had a color image, you could repeat this with all channels here.
            # Find sub-pixels data:
            old_x = new_x * inv_scale_factor_x
            old_y = new_y * inv_scale_factor_y
            x_fraction = old_x - math.floor(old_x)
            y_fraction = old_y - math.floor(old_y)

            # Sample four neighboring pixels:
            left_upper = image_matrix[math.floor(old_y), math.floor(old_x)]
            right_upper = image_matrix[math.floor(old_y), min(image_matrix.shape[1] - 1, math.ceil(old_x))]
            left_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), math.floor(old_x)]
            right_lower = image_matrix[
                min(image_matrix.shape[0] - 1, math.ceil(old_y)), min(image_matrix.shape[1] - 1, math.ceil(old_x))]

            # Interpolate horizontally:
            blend_top = (right_upper * x_fraction) + (left_upper * (1.0 - x_fraction))
            blend_bottom = (right_lower * x_fraction) + (left_lower * (1.0 - x_fraction))
            # Interpolate vertically:
            final_blend = (blend_top * y_fraction) + (blend_bottom * (1.0 - y_fraction))
            output_image[new_y, new_x] = final_blend

    return output_image


def sp_noise(image, prob):
    """
    Add salt and pepper noise to image
    prob: Probability of the noise
    """
    output = image.copy()
    if len(image.shape) == 2:
        black = 0
        white = 255
    else:
        color_space = image.shape[2]
        if color_space == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    return output


class DatasetAugmenter:
    def __init__(self, images_dir: str, output_dir: str):
        self.original_images_path_list = os.listdir(images_dir)
        self.images_dir = images_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def run(self):
        for image_name in self.original_images_path_list:
            try:
                image = cv2.imread(f"{self.images_dir}/{image_name}", 0)

                noise_threshold = random.choice([0.1, 0.3])
                noise_image = sp_noise(image, noise_threshold)

                print(f"[INFO] Reading and augmenting image: {image_name}")
                image_resized = resize_linear(noise_image, new_height=opt.img_size, new_width=opt.img_size)
                cv2.imwrite(f"{self.output_dir}/noise_image_{image_name.replace('.png', '')}.png", image_resized)

                # Applying augmentation
                for i in tqdm(range(500)):
                    augmented_image = transform(image=image)['image']
                    image_resized = resize_linear(augmented_image, new_height=opt.img_size, new_width=opt.img_size)
                    cv2.imwrite(f"{self.output_dir}/augmented_{image_name.replace('.png', '')}_{i}.png",
                                image_resized)

            except AttributeError as e:
                print(f"Image {image_name} error: {e.args}.")
                pass

print("[INFO] Loading and augmenting training images...")
augmenter = DatasetAugmenter(images_dir=opt.ti_folder,
                             output_dir=opt.output_folder)
print("[INFO] Successfully loaded images...")
print("[INFO] Applying augmentation...")
augmenter.run()