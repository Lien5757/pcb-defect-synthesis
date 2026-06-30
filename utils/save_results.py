import os
import numpy as np
import logging
from PIL import Image
from datetime import datetime
from utils.image_utils import combine_image_grid_batch, combine_image_with_prompt

logger = logging.getLogger(__name__)

def save_inpainted_results(base_batch, mask_batch, result_batch, prompts, save_dir, batch_idx, data_name, save_mode='grid'):
    os.makedirs(save_dir, exist_ok=True)

    if save_mode == 'grid':
        combined = combine_image_grid_batch([
            [img.resize((256, 256)) for img in base_batch],
            [img.resize((256, 256)) for img in mask_batch],
            [img.resize((256, 256)) for img in result_batch],
        ])
        combined.save(os.path.join(save_dir, f"batch_{batch_idx}.png"))
        logger.info(f"Saved: batch_{batch_idx}.png")

    elif save_mode == 'individual':  # individual save mode
        for k, img in enumerate(result_batch):
            img_np = np.array(img)
            if np.max(img_np) == 0:
                logger.warning(f"Inpainted image {k} is completely black")
                continue
            
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            save_path = setting_save_path(data_name, prompt, timestamp, batch_idx, k, save_dir)

            img.save(save_path)
            logger.info(f'Saved: {save_path}')
    
    elif save_mode == 'individual_with_3':
        for k, img in enumerate(result_batch):
            img_np = np.array(img)
            if np.max(img_np) == 0:
                logger.warning(f"Inpainted image {k} is completely black")
                continue

            prompt = prompts[k]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # single image
            save_path, filename = setting_save_path(data_name, prompt, timestamp, batch_idx, k, save_dir)
            img.save(save_path)
            logger.info(f'Inpainted result saved: {save_path}')

            # combined image
            combined_dir = os.path.join(save_dir, 'combine_grid')
            os.makedirs(combined_dir, exist_ok=True)

            combined = combine_image_with_prompt(base_batch[k], mask_batch[k], img, prompt)
            save_grid_path = os.path.join(combined_dir, filename)
            combined.save(save_grid_path)
            logger.info(f'Paired image saved: {save_grid_path}')

        # Batch grid
        combined = combine_image_grid_batch([
                [img.resize((256, 256)) for img in base_batch],
                [img.resize((256, 256)) for img in mask_batch],
                [img.resize((256, 256)) for img in result_batch],
        ])

        grid_dir = os.path.join(save_dir, 'batch_grid')
        os.makedirs(grid_dir, exist_ok=True)

        combined.save(os.path.join(grid_dir, f"batch_{batch_idx}.png"))
        logger.info(f"Saved: batch_{batch_idx}.png")
    
    elif save_mode == 'individual_with_4': # + Mask
        for k, img in enumerate(result_batch):
            img_np = np.array(img)
            if np.max(img_np) == 0:
                logger.warning(f"Inpainted image {k} is completely black")
                continue

            prompt = prompts[k]
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # single image
            save_path, filename = setting_save_path(data_name, prompt, timestamp, batch_idx, k, save_dir)
            img.save(save_path)
            logger.info(f'Saved: {save_path}')

            # combined image
            combined_dir = os.path.join(save_dir, 'combine_grid')
            os.makedirs(combined_dir, exist_ok=True)

            combined = combine_image_with_prompt(base_batch[k], mask_batch[k], img, prompt)
            save_grid_path = os.path.join(combined_dir, filename)
            combined.save(save_grid_path)
            logger.info(f'Saved: {save_grid_path}')

            # mask image
            mask_dir = os.path.join(save_dir, 'masks')
            os.makedirs(mask_dir, exist_ok=True) 
            mask_img = mask_batch[k]
            save_mask_path = os.path.join(mask_dir, filename)
            mask_img.save(save_mask_path)           

        # Batch grid
        combined = combine_image_grid_batch([
                [img.resize((256, 256)) for img in base_batch],
                [img.resize((256, 256)) for img in mask_batch],
                [img.resize((256, 256)) for img in result_batch],
        ])

        grid_dir = os.path.join(save_dir, 'batch_grid')
        os.makedirs(grid_dir, exist_ok=True)

        combined.save(os.path.join(grid_dir, f"batch_{batch_idx}.png"))
        logger.info(f"Saved: batch_{batch_idx}.png")

def setting_save_path(data_name, prompt, timestamp, batch_idx, k, save_dir):
    if data_name == 'exp5' or 'exp5_v2' or'exp3':
        prompt_shade = prompt.split(' ')[1]
        prompt_color = prompt.split(' ')[2]
        filename = f'inpaint_{prompt_shade}_{prompt_color}_{batch_idx}_{k}_{timestamp}.png'
        
        save_path = os.path.join(save_dir, filename)

    else:
        defect_type_1 = prompt.split(' ')[0]
        defect_type_2 = prompt.split(' ')[1]
        filename = f'inpaint_{defect_type_1}_{defect_type_2}_{batch_idx}_{k}_{timestamp}.png'

        os.makedirs(os.path.join(save_dir, 'inpainted_results'), exist_ok=True)
        save_path = os.path.join(save_dir, 'inpainted_results', filename)

    return save_path, filename
