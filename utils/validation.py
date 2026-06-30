"""Validation utilities for training and inference."""

import os
from typing import List
from utils.datasets import InpaintingDataset


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_training_config(data_dir: str) -> None:
    """Validate training configuration and data directory structure.

    Args:
        data_dir: Path to training data directory.

    Raises:
        ValidationError: If validation fails with meaningful error message.
    """
    if not isinstance(data_dir, str):
        raise ValidationError(f"data_dir must be a string, got {type(data_dir)}")

    # Check data_dir exists
    if not os.path.exists(data_dir):
        raise ValidationError(
            f"Data directory does not exist: {data_dir}\n"
            f"Please ensure the path is correct and the directory exists."
        )

    if not os.path.isdir(data_dir):
        raise ValidationError(f"data_dir is not a directory: {data_dir}")

    # Check required subdirectories
    required_subdirs = ["images", "masks", "texts"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            raise ValidationError(
                f"Required subdirectory missing: {subdir}/\n"
                f"Expected path: {subdir_path}\n"
                f"Data structure should be: {data_dir}/(images,masks,texts)/(class_name)/(files)"
            )

        if not os.path.isdir(subdir_path):
            raise ValidationError(f"{subdir} exists but is not a directory: {subdir_path}")

        # Check if subdirectory is empty
        if not os.listdir(subdir_path):
            raise ValidationError(
                f"Subdirectory is empty: {subdir}/\n"
                f"Expected path: {subdir_path}"
            )


def validate_dataset(data_dir: str) -> InpaintingDataset:
    """Validate and load training dataset.

    Args:
        data_dir: Path to training data directory.

    Returns:
        Loaded InpaintingDataset.

    Raises:
        ValidationError: If dataset is invalid or empty.
    """
    validate_training_config(data_dir)

    try:
        dataset = InpaintingDataset(base_dir=data_dir, transform=None)
    except Exception as e:
        raise ValidationError(
            f"Failed to load dataset from {data_dir}\n"
            f"Error: {str(e)}"
        )

    if len(dataset) == 0:
        raise ValidationError(
            f"Dataset is empty after loading from {data_dir}\n"
            f"Check that images, masks, and texts are properly aligned:\n"
            f"  - images/{{class_name}}/{{name}}.png\n"
            f"  - masks/{{class_name}}/{{name}}.png\n"
            f"  - texts/{{class_name}}/{{name}}.txt"
        )

    return dataset


def validate_model_path(model_path: str) -> None:
    """Validate model checkpoint path.

    Args:
        model_path: Path to model checkpoint file.

    Raises:
        ValidationError: If model path is invalid.
    """
    if not isinstance(model_path, str):
        raise ValidationError(f"model_path must be a string, got {type(model_path)}")

    if not os.path.exists(model_path):
        raise ValidationError(
            f"Model checkpoint not found: {model_path}\n"
            f"Please check the path and ensure the file exists."
        )

    if not os.path.isfile(model_path):
        raise ValidationError(f"model_path is not a file: {model_path}")

    # Check file extension
    valid_extensions = ('.pt', '.pth')
    if not model_path.endswith(valid_extensions):
        raise ValidationError(
            f"Model checkpoint has invalid extension: {model_path}\n"
            f"Expected .pt or .pth files"
        )

    # Check file is readable and not corrupted
    try:
        file_size = os.path.getsize(model_path)
        if file_size == 0:
            raise ValidationError(f"Model checkpoint is empty: {model_path}")
        if file_size < 1024:  # Less than 1KB is suspicious for a model
            raise ValidationError(
                f"Model checkpoint is suspiciously small ({file_size} bytes): {model_path}"
            )
    except OSError as e:
        raise ValidationError(f"Cannot access model checkpoint: {model_path}\n{str(e)}")


def validate_prediction_dirs(base_dir: str, mask_dir: str) -> None:
    """Validate inference input directories.

    Args:
        base_dir: Directory containing base images.
        mask_dir: Directory containing mask images.

    Raises:
        ValidationError: If directories are invalid or empty.
    """
    for dir_path, dir_name in [(base_dir, "base"), (mask_dir, "mask")]:
        if not isinstance(dir_path, str):
            raise ValidationError(f"{dir_name}_dir must be a string, got {type(dir_path)}")

        if not os.path.exists(dir_path):
            raise ValidationError(
                f"{dir_name.capitalize()} directory does not exist: {dir_path}\n"
                f"Please ensure the path is correct."
            )

        if not os.path.isdir(dir_path):
            raise ValidationError(f"{dir_name}_dir is not a directory: {dir_path}")

        # Check directory contains images
        image_files = _get_image_files(dir_path)
        if not image_files:
            raise ValidationError(
                f"No image files found in {dir_name} directory: {dir_path}\n"
                f"Supported formats: .png, .jpg, .jpeg, .bmp"
            )


def validate_batch_size(batch_size: int, dataset_size: int) -> None:
    """Validate batch size is reasonable for dataset.

    Args:
        batch_size: Batch size value.
        dataset_size: Total number of samples in dataset.

    Raises:
        ValidationError: If batch size is invalid.
    """
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValidationError(f"batch_size must be a positive integer, got {batch_size}")

    if batch_size > dataset_size:
        raise ValidationError(
            f"batch_size ({batch_size}) is larger than dataset size ({dataset_size})\n"
            f"Reduce batch_size or check your dataset."
        )


def _get_image_files(directory: str) -> List[str]:
    """Get list of image files in directory.

    Args:
        directory: Directory path.

    Returns:
        List of image file paths.
    """
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    try:
        files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith(supported_extensions) and os.path.isfile(os.path.join(directory, f))
        ]
        return files
    except OSError as e:
        raise ValidationError(f"Cannot read directory {directory}: {str(e)}")
