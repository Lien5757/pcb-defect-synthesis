import torch
import logging
from diffusers import (StableDiffusionInpaintPipeline, DDIMScheduler, EulerAncestralDiscreteScheduler,
                        LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler)

logger = logging.getLogger(__name__)

def load_model(checkpoint_path, device='cuda', scheduler_type="DDIM"):
    pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)
    if scheduler_type == "EulerA":
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DDIM":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "LMS":
        pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "PNDM":
        pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    elif scheduler_type == "DPM-Solver":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    pipe.unet.load_state_dict(checkpoint["unet"] if "unet" in checkpoint else checkpoint)
    logger.info(f"Load model: {checkpoint_path}")
    return pipe