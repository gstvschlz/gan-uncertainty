"""
Applying covariance to training images
"""
import math
import os
import random

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

original_images_path_list = os.listdir("dataset/training_images/")
os.makedirs("dataset/generated_images/", exist_ok=True)


def resize_linear(image_matrix, new_height:int, new_width:int):
    """Perform a pure-numpy linear-resampled resize of an image."""
    output_image = np.zeros((new_height, new_width), dtype=image_matrix.dtype)
    original_height, original_width = image_matrix.shape
    inv_scale_factor_y = original_height/new_height
    inv_scale_factor_x = original_width/new_width

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
            right_lower = image_matrix[min(image_matrix.shape[0] - 1, math.ceil(old_y)), min(image_matrix.shape[1] - 1, math.ceil(old_x))]

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


transform = A.Compose([
    A.RandomRotate90(),
    A.Flip(),
    A.Transpose(),
    A.OneOf([
        A.GaussianBlur(),
        A.GaussNoise(),
    ], p=0.3),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.Blur(blur_limit=3, p=0.1),
    ], p=0.6),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    A.OneOf([
        A.OpticalDistortion(p=0.3),
        A.GridDistortion(p=.1),
        A.PiecewiseAffine(p=0.3),
    ], p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.Sharpen(),
        A.Emboss(),
        A.RandomBrightnessContrast(),
    ], p=0.4),
    A.HueSaturationValue(p=0.4),
])

ignore_list = []

for image_name in tqdm(original_images_path_list):
    if image_name not in ignore_list:
        try:
            image = cv2.imread(f"dataset/training_images/{image_name}", 0)

            noise_threshold = random.choice([0.1, 0.3])
            noise_image = sp_noise(image, noise_threshold)

            print(f"Reading and augmenting image: {image_name}")
            image_resized = resize_linear(noise_image, new_height=64, new_width=64)
            cv2.imwrite(f"dataset/generated_images/noise_image_{image_name.replace('.png', '')}.png", image_resized)
            # Applying augmentation
            for i in tqdm(range(50)):
                augmented_image = transform(image=image)['image']
                image_resized = resize_linear(augmented_image, new_height=64, new_width=64)
                cv2.imwrite(f"dataset/generated_images/augmented_{image_name.replace('.png', '')}_{i}.png", image_resized)

        except AttributeError or cv2.error as e:
            print(f"Image {image_name} error: {e.args}.")
            pass

