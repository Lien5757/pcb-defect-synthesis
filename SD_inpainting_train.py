import os
import time
import logging
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from transformers import get_linear_schedule_with_warmup

# Import utils
from utils.plot_utils import show_batch_images_and_masks, plot_loss_live, plot_lr
from data_loader.loader import load_train_data
from config import TrainingConfig

class StableDiffusionInpainterTrainer:
    def __init__(self, config: TrainingConfig):
        today_str = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        self.config = config
        self.data_dir = config.data_dir
        self.project_name = f"{config.project_name}_{today_str}"
        self.is_transform = config.is_transform
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.min_delta = config.min_delta
        self.use_warmup = config.use_warmup
        self.warmup_ratio = config.warmup_ratio
        self.use_weighted_sampler = config.use_weighted_sampler
        self.save_interval = config.save_interval

        # Save checkpoint path
        self.save_dir = os.path.join('checkpoints', self.project_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.resume_path = os.path.join(self.save_dir, "latest.pt")  

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = self._load_pipeline()
        self.noise_scheduler = self._create_noise_scheduler()
        self.optimizer = AdamW(self.pipe.unet.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler = torch.GradScaler(device=self.device)
        self.best_loss = float("inf")
        self.best_model_path = os.path.join(self.save_dir, "best_model.pt")
        self.loss_history = []
        self.lr_history = []
        self.start_epoch = 0
        self.no_improve_epochs = 0
        self.logger = self._setup_logger()

        if self.resume_path:
            self._load_checkpoint(self.resume_path)

    def _load_pipeline(self):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32
        )
        pipe.to(self.device)
        pipe.vae.eval()
        pipe.text_encoder.eval()
        for param in pipe.vae.parameters():
            param.requires_grad = False
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False
        return pipe

    def _create_noise_scheduler(self):
        return DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1
        )

    def _load_checkpoint(self, path):
        if not os.path.exists(path):
            self.logger.info(f"No checkpoint found at {path}. Starting fresh.")
            return

        checkpoint = torch.load(path, map_location=self.device)
        self.pipe.unet.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("best_loss", self.best_loss)
        self.logger.info(f"Resumed training from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.pipe.unet.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_loss": self.best_loss
        }
        torch.save(checkpoint, os.path.join(self.save_dir, "latest.pt"))
    
    def _setup_logger(self):
        log_path = os.path.join(self.save_dir, "training.log")
        logger = logging.getLogger(self.project_name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')  # define first

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # Avoid duplicate file handlers
        if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger


    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Data Dir: {self.data_dir}")
        self.logger.info(f"Project: {self.project_name}")
        self.logger.info(f"Num Epochs: {self.num_epochs}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Learning Rate: {self.lr} Weight Decay: {self.weight_decay}")
        self.logger.info(f"Use Warmup: {self.use_warmup} Warmup Ratio: {self.warmup_ratio}")
        self.logger.info(f"Use Weighted Sampler: {self.use_weighted_sampler}")

        dataset, dataloader, sample_weights = load_train_data(self.data_dir, self.batch_size, self.is_transform, self.use_weighted_sampler)
        self.logger.info(f"Sample weights:{sample_weights}")

        # Warmup setup
        total_steps = (len(dataset) // self.batch_size) * self.num_epochs
        if self.use_warmup:
            warmup_steps = int(self.warmup_ratio * total_steps)
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            # Use a flat scheduler that keeps LR constant
            scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)


        for epoch in range(self.start_epoch, self.num_epochs):
            self.pipe.unet.train()
            epoch_loss = 0
            start_time = time.time()

            for idx, batch in enumerate(dataloader):
                defect_images = batch["defect_image"].to(self.device, dtype=torch.float32)
                masks = batch["mask"].to(self.device, dtype=torch.float32)
                masked_images = batch["masked_image"].to(self.device, dtype=torch.float32)
                prompts = batch["prompt"]

                if epoch == 0 and idx < 10:
                    show_batch_images_and_masks(defect_images, masks, masked_images, wait_time=100)
                elif epoch == 0 and idx == 10:
                    import cv2
                    cv2.destroyAllWindows()

                text_inputs = self.pipe.tokenizer(prompts, padding="max_length", return_tensors="pt").input_ids.to(self.device)
                text_embeddings = self.pipe.text_encoder(text_inputs)[0].to(dtype=torch.float32)

                with torch.no_grad():
                    defect_latents = self.pipe.vae.encode(defect_images).latent_dist.sample().to(dtype=torch.float32)
                    defect_latents *= self.pipe.vae.config.scaling_factor
                    masked_latents = self.pipe.vae.encode(masked_images).latent_dist.sample().to(dtype=torch.float32)
                    masked_latents *= self.pipe.vae.config.scaling_factor

                mask = F.interpolate(masks, size=(64, 64), mode="nearest").to(dtype=torch.float32)
                timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (defect_latents.shape[0],), device=self.device).long()
                noise = torch.randn_like(defect_latents)
                noisy_latents = self.noise_scheduler.add_noise(defect_latents, noise, timesteps)
                unet_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)

                self.optimizer.zero_grad()
                with torch.autocast(device_type=self.device, dtype=torch.float16):
                    noise_pred = self.pipe.unet(unet_input, timesteps, encoder_hidden_states=text_embeddings).sample

                    # Match dtypes
                    mask = mask.to(noise_pred.dtype) # torch.float16
                    noise = noise.to(noise_pred.dtype) # torch.float16

                    ## Loss(consider unmasked region)
                    # loss_masked = F.mse_loss(noise_pred * mask, noise * mask) / (mask.mean() + 1e-6)
                    # loss_global = F.mse_loss(noise_pred, noise)
                    # loss = 0.9 * loss_masked + 0.1 * loss_global

                    ## Loss
                    loss = F.mse_loss(noise_pred * mask, noise * mask) / (mask.mean() + 1e-6)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scheduler.step()
                self.lr_history.append(scheduler.get_last_lr()[0])
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            self.loss_history.append(avg_loss)

            end_time = time.time()
            epoch_time = end_time - start_time
            self.logger.info(f"Epoch [{epoch+1}/{self.num_epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

            if (epoch + 1) % self.save_interval == 0:
                self._save_checkpoint(epoch)
                self.logger.info(f"Checkpoint saved at epoch {epoch+1}")

            if avg_loss < self.best_loss - self.min_delta:
                self.best_loss = avg_loss
                self.no_improve_epochs = 0
                torch.save(self.pipe.unet.state_dict(), self.best_model_path)
                self.logger.info(f"New best model saved with loss {self.best_loss:.4f}")
            else:
                self.no_improve_epochs += 1
                self.logger.info(f"No improvement for {self.no_improve_epochs} epoch(s)")

            plot_loss_live(epoch, self.loss_history, plot_interval=1, save_dir=self.save_dir)

        torch.save(self.pipe.unet.state_dict(), os.path.join(self.save_dir, "final_model.pt"))
        plot_lr(self.lr_history, self.save_dir)
        self.logger.info("Training Complete.")

def parse_args():
    parser = argparse.ArgumentParser(description="Train SD Inpainting model")
    parser.add_argument("--config", type=str, default=None, help="Path to config JSON file")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to training data")
    parser.add_argument("--project_name", type=str, default=None, help="Project name")
    parser.add_argument("--is_transform", action="store_true", help="Enable data augmentation")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--use_warmup", action="store_true", help="Enable warmup")
    parser.add_argument("--warmup_ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--use_weighted_sampler", action="store_true", help="Enable weighted sampler")
    parser.add_argument("--min_delta", type=float, default=None, help="Min delta for loss tracking")
    parser.add_argument("--save_interval", type=int, default=None, help="Checkpoint save interval")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load config from file if provided
    if args.config:
        config = TrainingConfig.from_json(args.config)
    else:
        # Use default config
        config = TrainingConfig(data_dir=r'datasets\train\exp1')

    # Override with CLI args
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    if cli_args:
        config.update_from_args(**cli_args)

    trainer = StableDiffusionInpainterTrainer(config)
    trainer.train()
