import os
import numpy as np
import logging
from PIL import Image
from datetime import datetime
from utils.image_utils import combine_image_grid_batch, combine_image_with_prompt
from utils.prompt_loader import get_class_name_from_prompt

logger = logging.getLogger(__name__)

def save_inpainted_results(base_batch, mask_batch, result_batch, prompts, save_dir, batch_idx, data_name, save_mode='individual_with_3'):
    """Save inpainted results with individual images and paired grids.

    Args:
        base_batch: List of base images.
        mask_batch: List of mask images.
        result_batch: List of inpainted result images.
        prompts: List of prompts corresponding to images.
        save_dir: Directory to save results.
        batch_idx: Batch index.
        data_name: Dataset name for class name lookup.
        save_mode: Save mode (currently only 'individual_with_3' supported).

    Raises:
        ValueError: If save_mode is not 'individual_with_3' or prompt not found.
    """
    if save_mode != 'individual_with_3':
        logger.warning(f"save_mode '{save_mode}' is deprecated. Using 'individual_with_3'.")

    os.makedirs(save_dir, exist_ok=True)

    # Save individual results with paired grids
    for k, img in enumerate(result_batch):
        img_np = np.array(img)
        if np.max(img_np) == 0:
            logger.warning(f"Inpainted image {k} is completely black")
            continue

        prompt = prompts[k]
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Get class name from prompt
        try:
            class_name = get_class_name_from_prompt(data_name, prompt)
        except ValueError as e:
            logger.error(f"Cannot find class for prompt: {prompt}\nError: {str(e)}")
            raise

        # Save single inpainted image
        save_path, filename = _save_individual_image(img, class_name, batch_idx, k, timestamp, save_dir)
        logger.info(f'Inpainted result saved: {save_path}')

        # Save paired grid (base + mask + result)
        combined = combine_image_with_prompt(base_batch[k], mask_batch[k], img, prompt)
        combined_dir = os.path.join(save_dir, 'combine_grid')
        os.makedirs(combined_dir, exist_ok=True)
        save_grid_path = os.path.join(combined_dir, filename)
        combined.save(save_grid_path)
        logger.info(f'Paired image saved: {save_grid_path}')

    # Save batch grid
    combined = combine_image_grid_batch([
        [img.resize((256, 256)) for img in base_batch],
        [img.resize((256, 256)) for img in mask_batch],
        [img.resize((256, 256)) for img in result_batch],
    ])

    grid_dir = os.path.join(save_dir, 'batch_grid')
    os.makedirs(grid_dir, exist_ok=True)

    combined.save(os.path.join(grid_dir, f"batch_{batch_idx}.png"))
    logger.info(f"Batch grid saved: batch_{batch_idx}.png")


def _save_individual_image(img, class_name, batch_idx, k, timestamp, save_dir):
    """Save individual inpainted image with class-based naming.

    Args:
        img: PIL Image to save.
        class_name: Class name from prompts.json (e.g., '0_dark_blue').
        batch_idx: Batch index.
        k: Image index within batch.
        timestamp: Timestamp string.
        save_dir: Directory to save image.

    Returns:
        Tuple of (save_path, filename).
    """
    filename = f'inpaint_{class_name}_{batch_idx}_{k}_{timestamp}.png'
    save_path = os.path.join(save_dir, filename)
    img.save(save_path)
    return save_path, filename