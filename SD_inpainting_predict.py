import os
import torch
import numpy as np
import logging
from PIL import Image
from datetime import datetime
from typing import List, Optional
import cv2
import random

# Import utils
from utils.augmentation import random_flip_rotate_pil
from utils.mask_utils import random_transform
from utils.prompt_utils import set_prompts, set_prompts_given
from utils.load_model import load_model
from utils.predict import predict_batch
from utils.save_results import save_inpainted_results
from utils.validation import validate_model_path, validate_prediction_dirs, ValidationError
from config.enums import PromptMode


class Inpainter:
    """Run inference with trained SD Inpainting model."""

    def __init__(
        self,
        model_path: str,
        data_name: str = 'exp1',
        enable_aug_on_mask: bool = True,
        enable_aug_on_base: bool = True,
        scheduler_type: str = 'DDIM',
        device: Optional[str] = None
    ) -> None:
        """Initialize inpainter with model and configuration.

        Args:
            model_path: Path to trained model checkpoint.
            data_name: Dataset name for prompt selection.
            enable_aug_on_mask: Apply augmentation to masks.
            enable_aug_on_base: Apply augmentation to base images.
            scheduler_type: Noise scheduler type ('DDIM' or others).
            device: Device to use ('cuda:0' or 'cpu'). Auto-detected if None.

        Raises:
            ValidationError: If model path is invalid.
        """
        # Validate model path before loading
        try:
            validate_model_path(model_path)
        except ValidationError as e:
            raise ValidationError(f"Model validation failed:\n{str(e)}")

        self.model_path = model_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self.pipe = load_model(model_path, device=self.device, scheduler_type=scheduler_type)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model from {model_path}\n"
                f"Error: {str(e)}"
            )
        self.data_name = data_name
        self.enable_aug_on_mask = enable_aug_on_mask
        self.enable_aug_on_base = enable_aug_on_base

        project_name = os.path.basename(os.path.dirname(model_path))
        model_filename = os.path.basename(model_path)

        model_type = os.path.splitext(model_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        self.save_dir = os.path.join('output', project_name, f"{model_type}_{timestamp}")
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logger for inference process.

        Returns:
            Configured logger instance.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        log_path = os.path.join(self.save_dir, "inference.log")
        logger = logging.getLogger("Inpainter")
        logger.setLevel(logging.INFO)

        # Clear existing handlers to avoid stale file paths
        logger.handlers.clear()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    def load_images(self, base_dir: str, mask_dir: str) -> tuple[List[str], List[str]]:
        """Load base and mask images, matching them by count.

        Args:
            base_dir: Directory containing base images.
            mask_dir: Directory containing mask images.

        Returns:
            Tuple of (base_image_paths, mask_image_paths) with matched lengths.

        Raises:
            ValueError: If no images found in either directory.
        """
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        try:
            base_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                          if f.lower().endswith(supported_extensions)]
            mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                          if f.lower().endswith(supported_extensions)]
        except OSError as e:
            raise ValueError(f"Error reading image directories: {str(e)}")

        if not base_images:
            raise ValueError(
                f"No base images found in: {base_dir}\n"
                f"Supported formats: {supported_extensions}"
            )

        if not mask_images:
            raise ValueError(
                f"No mask images found in: {mask_dir}\n"
                f"Supported formats: {supported_extensions}"
            )

        if len(mask_images) > len(base_images):
            match_mask_images = random.sample(mask_images, len(base_images))
        elif len(mask_images) < len(base_images):
            extra_masks = random.choices(mask_images, k=len(base_images) - len(mask_images))
            match_mask_images = mask_images + extra_masks
        else:
            match_mask_images = mask_images

        return base_images, match_mask_images

    def process_mask(self, mask_path: str) -> Image.Image:
        """Load and binarize mask image.

        Args:
            mask_path: Path to mask image.

        Returns:
            Binarized PIL Image.

        Raises:
            ValueError: If mask cannot be loaded or is invalid.
        """
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to read mask image (cv2.imread returned None)")
        except Exception as e:
            raise ValueError(f"Cannot read mask image from {mask_path}: {str(e)}")

        try:
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
            return Image.fromarray(mask)
        except Exception as e:
            raise ValueError(f"Cannot process mask image: {str(e)}")

    def run(
        self,
        base_dir: str,
        mask_dir: str,
        prompt_mode: PromptMode | str = PromptMode.MULTI,
        prompt: Optional[str] = None,
        batch_size: int = 4,
        target_total: Optional[int] = None
    ) -> None:
        """Run inference on base images with masks and save results.

        Args:
            base_dir: Directory of base images.
            mask_dir: Directory of mask images.
            prompt_mode: PromptMode.MULTI for dataset prompts or PromptMode.SINGLE for custom.
            prompt: Custom prompt text (required if prompt_mode=PromptMode.SINGLE).
            batch_size: Number of images per batch.
            target_total: Total images to generate (replicates if needed).

        Raises:
            ValidationError: If input directories are invalid.
            ValueError: If prompt_mode is SINGLE but prompt is not provided.
        """
        if isinstance(prompt_mode, str):
            prompt_mode = PromptMode(prompt_mode)

        # Validate input directories
        try:
            validate_prediction_dirs(base_dir, mask_dir)
        except ValidationError as e:
            raise ValidationError(f"Input directory validation failed:\n{str(e)}")

        # Validate prompt configuration
        if prompt_mode == PromptMode.SINGLE and not prompt:
            raise ValueError(
                "prompt_mode is PromptMode.SINGLE but prompt parameter is not provided.\n"
                "Please provide a custom prompt text."
            )

        os.makedirs(self.save_dir, exist_ok=True)

        base_images, match_mask_images = self.load_images(base_dir, mask_dir)

        if target_total is not None and target_total > len(base_images):
            extra_count = target_total - len(base_images)
            extra_bases = random.choices(base_images, k=extra_count)
            extra_masks = random.choices(match_mask_images, k=extra_count)
            base_images += extra_bases
            match_mask_images += extra_masks

        elif target_total is not None:
            base_images = base_images[:target_total]
            match_mask_images = match_mask_images[:target_total]

        total_batches = (len(base_images) + batch_size - 1) // batch_size
        self.logger.info(f"Starting inference on {len(base_images)} images ({total_batches} batches)")

        idx = 0

        for i in range(0, len(base_images), batch_size):
            base_batch_paths = base_images[i:i+batch_size]
            mask_batch_paths = match_mask_images[i:i+batch_size]

            try:
                self.logger.info(f"Processing batch {idx + 1}/{total_batches} ({len(base_batch_paths)} images)")

                # Load base images
                base_batch = []
                for p in base_batch_paths:
                    try:
                        base_batch.append(Image.open(p).convert("RGB"))
                    except Exception as e:
                        raise ValueError(f"Cannot load base image from {p}: {str(e)}")

                if self.enable_aug_on_base:
                    base_batch = [random_flip_rotate_pil(img) for img in base_batch]

                # Load and process masks
                mask_batch = []
                for p in mask_batch_paths:
                    try:
                        mask_batch.append(self.process_mask(p))
                    except ValueError as e:
                        raise ValueError(f"Error processing mask {p}: {str(e)}")

                if self.enable_aug_on_mask:
                    mask_batch = [
                        Image.fromarray(random_transform(np.array(p)))
                        for p in mask_batch
                    ]

                if prompt_mode == PromptMode.MULTI:
                    prompts = set_prompts(batch_size=len(base_batch), data=self.data_name)
                elif prompt_mode == PromptMode.SINGLE:
                    prompts = set_prompts_given(batch_size=len(base_batch), prompt=prompt)

                results = predict_batch(self.pipe, base_batch, mask_batch, prompts, self.device)
                save_inpainted_results(base_batch, mask_batch, results, prompts, self.save_dir, idx + 1, data_name=self.data_name)

            except Exception as e:
                self.logger.error(f"Error processing batch {idx + 1}: {str(e)}")
                raise

            idx += 1

        self.logger.info(f'Inpainting process completed. Results saved to: {self.save_dir}')

if __name__ == "__main__":
    inpainter = Inpainter(
        model_path=r"checkpoints\exp1\best_model.pt",
        data_name="exp1",
        enable_aug_on_base=True,
        enable_aug_on_mask=True,
        scheduler_type="DDIM"
    )

    inpainter.run(
        base_dir=r"datasets\test\base",
        mask_dir=r"datasets\test\masks",
        prompt_mode=PromptMode.MULTI,
        batch_size=18,
    )