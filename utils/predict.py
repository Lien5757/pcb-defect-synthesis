import time
import torch
import logging

logger = logging.getLogger(__name__)

@torch.no_grad()
def predict(pipe, base_image, mask, prompt, device, num_inference_steps=50):
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    start_time = time.time()
    result = pipe(
        prompt=prompt,
        image=base_image,
        mask_image=mask,
        num_inference_steps=num_inference_steps
    )
    end_time = time.time()
    logger.info(f'Prediction time: {end_time-start_time}s')

    return result.images[0]

@torch.no_grad()
def predict_batch(pipe, base_images, masks, prompts, device, num_inference_steps=50):
    assert len(base_images) == len(masks) == len(prompts), "Input lists must be the same length"
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    start_time = time.time()
    results = pipe(
        prompt=prompts,
        image=base_images,
        mask_image=masks,
        num_inference_steps=num_inference_steps,
    )
    end_time = time.time()
    logger.info(f'Average prediction time: {(end_time - start_time) / len(base_images):.2f}s')
    return results.images