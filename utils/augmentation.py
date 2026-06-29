import random
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from torchvision import transforms
import torch


class ImageOnlyTransform:
    """Augmentation pipeline for inpainting data triplets (image, mask, masked_image)."""

    def __init__(self, crop_size: Tuple[int, int] = (448, 448), flip_prob: float = 0.5) -> None:
        """Initialize augmentation with crop and flip parameters.

        Args:
            crop_size: Size to crop images to.
            flip_prob: Probability of horizontal flip.
        """
        self.crop_size = crop_size
        self.flip_prob = flip_prob

    def __call__(self, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Apply augmentations consistently to all three images in sample.

        Args:
            sample: Dictionary with 'defect_image', 'mask', 'masked_image' tensors.

        Returns:
            Augmented sample dict or None if mask is too small.
        """
        img, mask, masked = sample["defect_image"], sample["mask"], sample["masked_image"]

        # Sample consistent crop
        i, j, h, w = transforms.RandomCrop.get_params(img, self.crop_size)
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        masked = transforms.functional.crop(masked, i, j, h, w)

        # Flip
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            masked = transforms.functional.hflip(masked)

        # Resize with proper interpolation
        img = transforms.functional.resize(img, [512, 512], interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, [512, 512], interpolation=transforms.InterpolationMode.NEAREST)
        masked = transforms.functional.resize(masked, [512, 512], interpolation=transforms.InterpolationMode.BILINEAR)

        # Optional: Skip sample if mask is too small
        if mask.sum() < 10:
            return None

        sample["defect_image"] = img
        sample["mask"] = mask
        sample["masked_image"] = masked
        return sample


def random_flip_rotate_pil(image: Image.Image) -> Image.Image:
    """
    Randomly flip (H/V) and rotate (90/180/270°) a PIL image.
    """
    flip_mode = random.choice(['H', 'V', None])
    if flip_mode == 'H':
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_mode == 'V':
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    rotation_angle = random.choice([90, 180, 270])
    image = image.rotate(rotation_angle, expand=True)

    return image
