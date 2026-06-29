import numpy as np
import random
import cv2

def random_transform(
    mask,
    max_translation=20,
    max_scale=0.2,
    max_rotation=30,
    direction="both",      # "both", "horizontal", "vertical", "none"
    allow_scaling=False,
    allow_rotation=False,
    isShow=False
):
    """
    Apply configurable random affine transformation to the mask.

    Args:
        mask (np.ndarray): The input binary mask.
        max_translation (int): Max translation in pixels (x/y direction).
        max_scale (float): Max scaling range (0.2 means ±20%).
        max_rotation (int): Max rotation angle in degrees.
        direction (str): 'both', 'horizontal', 'vertical', or 'none' for translation.
        allow_scaling (bool): Whether to apply random scaling.
        allow_rotation (bool): Whether to apply random rotation.
        isShow (bool): Whether to display the transformed mask for debugging.

    Returns:
        np.ndarray: Transformed mask.
    """
    h, w = mask.shape[:2]

    # Translation
    tx = ty = 0
    if direction == "both":
        tx = random.randint(-max_translation, max_translation)
        ty = random.randint(-max_translation, max_translation)
    elif direction == "horizontal":
        tx = random.randint(-max_translation, max_translation)
    elif direction == "vertical":
        ty = random.randint(-max_translation, max_translation)
    elif direction == "none":
        pass

    # Scaling
    scale = 1.0
    if allow_scaling:
        scale = 1 + random.uniform(-max_scale, max_scale)

    # Rotation
    angle = 0.0
    if allow_rotation:
        angle = random.uniform(-max_rotation, max_rotation)

    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty

    transformed_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)

    if np.max(transformed_mask) == 0:
        print("Warning: Transformed mask is completely black!")

    if isShow:
        cv2.imshow("Transformed Mask", transformed_mask)
        cv2.waitKey(1)

    return transformed_mask
