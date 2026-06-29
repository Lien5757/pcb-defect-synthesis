import os
import torch
import numpy as np
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


class Inpainter:
    """Run inference with trained SD Inpainting model."""

    def __init__(
        self,
        model_path: str,
        data_name: str = 'exp3',
        save_mode: str = 'grid',
        enable_aug_on_mask: bool = True,
        enable_aug_on_base: bool = True,
        scheduler_type: str = 'DDIM',
        device: Optional[str] = None
    ) -> None:
        """Initialize inpainter with model and configuration.

        Args:
            model_path: Path to trained model checkpoint.
            data_name: Dataset name for prompt selection.
            save_mode: Output saving mode ('grid' or 'individual').
            enable_aug_on_mask: Apply augmentation to masks.
            enable_aug_on_base: Apply augmentation to base images.
            scheduler_type: Noise scheduler type ('DDIM' or others).
            device: Device to use ('cuda:0' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pipe = load_model(model_path, device=self.device, scheduler_type=scheduler_type)
        self.data_name = data_name
        self.save_mode = save_mode
        self.enable_aug_on_mask = enable_aug_on_mask
        self.enable_aug_on_base = enable_aug_on_base

        project_name = os.path.basename(os.path.dirname(model_path))
        model_filename = os.path.basename(model_path)

        model_type = os.path.splitext(model_filename)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        self.save_dir = os.path.join('output', project_name, f"{model_type}_{timestamp}")

    def load_images(self, base_dir: str, mask_dir: str) -> tuple[List[str], List[str]]:
        """Load base and mask images, matching them by count.

        Args:
            base_dir: Directory containing base images.
            mask_dir: Directory containing mask images.

        Returns:
            Tuple of (base_image_paths, mask_image_paths) with matched lengths.
        """
        supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        base_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(supported_extensions)]
        mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(supported_extensions)]

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
        """
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        return Image.fromarray(mask)

    def run(
        self,
        base_dir: str,
        mask_dir: str,
        prompt_mode: str = 'multi',
        prompt: Optional[str] = None,
        batch_size: int = 4,
        target_total: Optional[int] = None
    ) -> None:
        """Run inference on base images with masks and save results.

        Args:
            base_dir: Directory of base images.
            mask_dir: Directory of mask images.
            prompt_mode: 'multi' for dataset prompts or 'single' for custom prompt.
            prompt: Custom prompt text (required if prompt_mode='single').
            batch_size: Number of images per batch.
            target_total: Total images to generate (replicates if needed).
        """
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

        idx = 0

        for i in range(0, len(base_images), batch_size):
            base_batch_paths = base_images[i:i+batch_size]
            mask_batch_paths = match_mask_images[i:i+batch_size]

            base_batch = [Image.open(p).convert("RGB") for p in base_batch_paths]
            if self.enable_aug_on_base:
                base_batch = [random_flip_rotate_pil(img) for img in base_batch]

            mask_batch = [self.process_mask(p) for p in mask_batch_paths]
            if self.enable_aug_on_mask:
                mask_batch = [
                    Image.fromarray(random_transform(np.array(p)))
                    for p in mask_batch
                ]

            if prompt_mode == 'multi':
                prompts = set_prompts(batch_size=len(base_batch), data=self.data_name)
            elif prompt_mode == 'single':
                prompts = set_prompts_given(batch_size=len(base_batch), prompt=prompt)
            results = predict_batch(self.pipe, base_batch, mask_batch, prompts, self.device)

            save_inpainted_results(base_batch, mask_batch, results, prompts, self.save_dir, idx + 1, data_name=self.data_name, save_mode=self.save_mode)
            idx += 1


        print('Inpainting process completed.')

if __name__ == "__main__":
    ## General Setup
    inpainter = Inpainter(
        model_path=r"checkpoints\exp5_fast_with_aug_ws_2\best_model.pt",
        data_name="exp5", # 透過data_name去控制prompt為哪個project的(set_prompts)
        save_mode="individual_with_3", # "individual" "individual_with_4"
        enable_aug_on_base=True, # False
        enable_aug_on_mask=True, # False
        scheduler_type="DDIM"
    )

    inpainter.run(
        base_dir=r"datasets\test\AOI\base\clean_images_20221118",
        mask_dir=r"datasets\test\AOI\masks\draw_mask_04_dry_films(exp5)",
        prompt_mode='multi', # 'multi' 'single'
        # prompt='A scratch defect on tray', # 刮傷這種遮罩類型不能混用的案例(上方選'single')
        batch_size=18, # 21:多跑幾次會out of memory
        # target_total=1000 # 當base影像不夠多的時候用的
    )

    ## For Tray datasets, containing different type of base image
    # data_dir = r'datasets\test\Tray\each_tray\base'
    # base_lists = os.listdir(data_dir)

    # remove_lists = ['Tray3_side1_P0-0-0','Tray3_side1_P0-0-1', 'Tray3_side1_P0-0-2','Tray3_side1_P0-0-3', 'Tray3_side1_P0-0-4','Tray3_side1_P0-0-5']
    # base_lists = [item for item in base_lists if item not in remove_lists]

    # for base_name in base_lists:
    #     inpainter.run(
    #         base_dir=fr"datasets\test\Tray\each_tray\base\{base_name}",
    #         mask_dir=r"datasets\test\Tray\general\draw_masks_Tray",
    #         save_dir=fr"output\exp6_Tray_no_aug\final_model(each)_2\{base_name}",
    #         prompt_mode='multi', # 'multi' 'single'
    #         # prompt='A scratch defect on tray',
    #         batch_size=12,
    #         target_total=100
    #     )

    