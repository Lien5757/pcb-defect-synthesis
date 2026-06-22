from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model, PeftModel
import torch
import os
import re
import cv2
import time
import random
import numpy as np
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
from torchsummary import summary

from utils.plot_utils import show_batch_images_and_masks, plot_loss_live
from utils.mask_utils import random_transform

class PCBInpaintingDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.transform = transform

        ## Get filename from defect image 
        self.image_files = []
        self.index_map = {}  # Record the mapping of index -> filename

        for f in os.listdir(base_dir):
            match = re.search(r"dry_film_defect_pcb_(\d+)\.png", f)  # Extract index from filename list
            if match:
                index = int(match.group(1))
                self.image_files.append((index, f))
                self.index_map[index] = f

        ## Sorted by image index
        self.image_files.sort()

        # Define transformation for images and masks separately
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        index, defect_filename = self.image_files[idx]

        # Filename of mask and prompt
        mask_filename = f"mask_{index:03d}.png"
        text_filename = f"defect_{index:03d}.txt"

        # Make sure file exist
        defect_image_path = os.path.join(self.base_dir, defect_filename)
        mask_image_path = os.path.join(self.base_dir, mask_filename)
        prompt_path = os.path.join(self.base_dir, text_filename)

        if not os.path.exists(mask_image_path) or not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Missing pair for index {index}: {defect_filename}, {mask_filename}, {text_filename}")

        ## Read defect image, mask and prompt 
        defect_image = Image.open(defect_image_path).convert("RGB")
        mask_image = Image.open(mask_image_path).convert("L")
        mask_image = mask_image.point(lambda p: 255 if p > 10 else 0)  # Binary threshold [0/255]
        with open(prompt_path, "r") as file:
            prompt = file.read().strip()

        ## Apply transformations
        defect_image = self.image_transform(defect_image)
        mask_image = self.mask_transform(mask_image) 

        ## Create masked image
        mask_binary = (mask_image < 0.5).float()  # Convert mask to binary [0 or 1]
        masked_image = defect_image * mask_binary  # Apply mask to image

        return {
            "defect_image": defect_image,  # (3, 512, 512)
            "mask": mask_image,  # (1, 512, 512)
            "masked_image": masked_image,  # (3, 512, 512)
            "prompt": prompt
        }

def check_model_structure(save_dir):
    ## Check device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ## Load pre-trained model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32
    ).to(device)

    ## check model structure
    with open(os.path.join(save_dir, "unet_structure.txt"), "w") as f:
        f.write("==== UNet Structure ====\n")
        f.write(str(pipe.unet))
        f.write("\n\n==== UNet Config ====\n")
        f.write(str(pipe.unet.config))
    
    with open(os.path.join(save_dir, "vae_structure.txt"), "w") as f:
        f.write("==== VAE Encoder Structure ====\n")
        f.write(str(pipe.vae.encode))
        f.write("\n\n==== VAE decoder Config ====\n")
        f.write(str(pipe.vae.decode))

    # Define LoRA config
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="UNET",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Key attention modules in U-Net
    )

    # Apply LoRA to the U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)
    with open(os.path.join(save_dir, "lora_unet_structure.txt"), "w") as f:
        f.write("==== LoRA-Injected UNet Structure ====\n")
        f.write(str(pipe.unet))
        f.write("\n\n==== LoRA-Injected UNet Config ====\n")
        f.write(str(pipe.unet.config))

def train_SD_inpaint_lora(data_dir, project='exp1', num_epochs=500, batch_size=1, lr=1e-6, save_interval=50):
    '''
    PCBInpaintingDataset增加masked_images，在pixel space計算過，然後再經過 VAE encoder，結果會更加穩定。
    torch.float32:16會使得loss為nan
    '''
    ## Save dir
    save_dir = os.path.join('checkpoints',project)
    os.makedirs(save_dir, exist_ok=True)

    ## Check device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ## Load pre-trained model
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32
    ).to(device)

    ## Define LoRA config
    'v1:Only attention LoRA'
    # lora_config = LoraConfig(
    #     r=4,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     bias="none",
    #     task_type="UNET",
    #     target_modules=[
    #         "to_q", "to_k", "to_v", "to_out.0"],  # Key attention modules in U-Net
    # )
    'v2:'
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type="UNET",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",        # cross/self-attn
            "conv1", "conv2",                          # in ResNet blocks
            "ff.net.0.proj"                            # optional: feedforward layers
        ]
    )

    # Apply LoRA to the U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Freeze Everything Except LoRA
    for name, param in pipe.unet.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    ## Datasets and Dataloader
    dataset = PCBInpaintingDataset(base_dir=data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    ## Optimizer
    optimizer = AdamW(pipe.unet.parameters(), lr=lr)

    # Initialize a list to store loss history
    loss_history = []

    ## Noise scheduler
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1
    )

    best_loss = float("inf")
    best_model_path = os.path.join(save_dir, "best_model")

    for epoch in range(num_epochs):
        pipe.unet.train()
        epoch_loss = 0

        for idx, batch in enumerate(dataloader):
            ## 1. Load data in batch
            defect_images = batch["defect_image"].to(device, dtype=torch.float32) # [batch, 3, 512, 512]
            masks = batch["mask"].to(device, dtype=torch.float32)
            masked_images = batch["masked_image"].to(device, dtype=torch.float32)
            prompts = batch["prompt"]
            # print(f"Image min: {defect_images.min().item()}, max: {defect_images.max().item()}") # [-1, 1]
            # print(f"Mask min: {masks.min().item()}, max: {masks.max().item()}") # [0, 1]

            ## Show the batch images and masks
            # show_batch_images_and_masks(defect_images, masks, masked_images)
            
            ## 2.1 Text to text embedding
            text_inputs = pipe.tokenizer(prompts, padding="max_length", return_tensors="pt").to(device)
            text_embeddings = pipe.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )[0].to(dtype=torch.float32)

            ## 2.2 Convert defect images to latent space
            with torch.no_grad():
                defect_latents = pipe.vae.encode(defect_images).latent_dist.sample().to(dtype=torch.float32)
                defect_latents = defect_latents * pipe.vae.config.scaling_factor # 0.18215 # [batch, 4, 64, 64]

                masked_latents = pipe.vae.encode(masked_images).latent_dist.sample().to(dtype=torch.float32)
                masked_latents = masked_latents * pipe.vae.config.scaling_factor

            # print(f"defect_latents min: {defect_latents.min().item()}, max: {defect_latents.max().item()}")
            # print(f"masked_latents min: {masked_latents.min().item()}, max: {masked_latents.max().item()}")
            # print(f"defect_latents shape: {defect_latents.shape}")  # (batch_size, 4, 64, 64)
            # print(f"masked_latents shape: {masked_latents.shape}")  # (batch_size, 4, 64, 64)
            if torch.isnan(defect_latents).any() or torch.isinf(defect_latents).any():
                print("Warning: defect_latents contains NaN or Inf!")

            
            ## 2.3 Resize mask to match latent space [batch, 1, 64, 64]
            mask = F.interpolate(masks, size=(64, 64), mode="nearest").to(dtype=torch.float32)
            # print(f"mask min: {mask.min().item()}, mask max: {mask.max().item()}")  # mask min: 0.0, mask max: 1.0
            # print(f"mask shape: {mask.shape}")  # (batch_size, 1, 64, 64)

            ## 3. Forward Diffusion: Add Noise to Latents
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(defect_latents, dtype=torch.float32)
            noisy_latents = noise_scheduler.add_noise(defect_latents, noise, timesteps)            

            ## Concatenate Inputs
            unet_input = torch.cat([noisy_latents, mask, masked_latents], dim=1)  # (batch_size, 9, 64, 64)
            # print(f"unet_input shape: {unet_input.shape}")  # (batch_size, 9, 64, 64)

            ## 4. Train UNet to Predict Noise
            noise_pred = pipe.unet(
                unet_input,
                timesteps,
                encoder_hidden_states=text_embeddings
            ).sample
            # print(f"noise_pred shape: {noise_pred.shape}") # (batch_size, 4, 64, 64)

            if torch.isnan(noise_pred).any():
                print("NaN detected in U-Net output! Checking input values:")
                print(f"min: {unet_input.min().item()}, max: {unet_input.max().item()}")
                print(f"mask min: {mask.min().item()}, max: {mask.max().item()}")
                print(f"text_embeddings min: {text_embeddings.min().item()}, max: {text_embeddings.max().item()}")
                noise_pred = torch.where(torch.isnan(noise_pred), torch.zeros_like(noise_pred), noise_pred)

            ## Compute Loss
            loss = F.mse_loss(noise_pred * mask, noise * mask) / (mask.mean() + 1e-6)
            # print(f"Loss: {loss.item()}")
            
            ## Back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Average epoch loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        loss_history.append(avg_loss)

        # Save periodically
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}")
            pipe.unet.save_pretrained(os.path.join(save_dir, f"model_epoch_{epoch + 1}"))
            print(f"Model saved at {checkpoint_path}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            pipe.unet.save_pretrained(best_model_path)
            print(f"Best model updated with loss {best_loss:.4f}, saved at {best_model_path}")

        # Plotting
        plot_loss_live(epoch, loss_history, plot_interval= 1, save_dir=save_dir)

    print('Training Done!')

@torch.no_grad()
def predict_batch_lora(lora_model_path, base_images, masks, prompts, device, num_inference_steps=50):
    assert len(base_images) == len(masks) == len(prompts), "base_images, masks, prompts 必須有相同長度"

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        use_safetensors=False
        ).to(device)

    ## Apply LoRA
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_model_path)

    # Set timesteps of Scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Inpainting (batch)
    start_time = time.time()
    results = pipe(
        prompt=prompts,         # 提示詞列表
        image=base_images,      # 多張影像 (List[PIL.Image])
        mask_image=masks,       # 多張 Mask (List[PIL.Image])
        num_inference_steps=num_inference_steps,
        # strength=0.5,             # denoising strength(0.3-0.5) # 1.0
        # guidance_scale=5.0        # 7.5
    )
    end_time = time.time()
    time_per_image = (end_time - start_time) / len(base_images)

    print(f'Average prediction time: {time_per_image:.2f}s')

    return results.images

def set_prompts(batch_size, data='exp2'):
    if data == 'exp3':
        # Generate all 6 prompt combinations
        shade_list = ['dark', 'light']
        color_list = ['blue', 'purple', 'yellow']
        prompt_list = [f"A {shade} {color} dry film residual defect." for shade in shade_list for color in color_list]

        # Repeat the 6 prompts evenly across the batch
        repeat_factor = batch_size // len(prompt_list)
        prompt_batch = prompt_list * repeat_factor

        # If not perfectly divisible, pad the remainder
        remainder = batch_size % len(prompt_list)
        if remainder > 0:
            prompt_batch += prompt_list[:remainder]
    elif data == 'exp2':
        prompt_batch = ['A blue dry film residual defect.'] * batch_size
    
    return prompt_batch

def inpaint_all_lora(base_dir, mask_dir, save_dir, lora_model_path, batch_size=4, data_name='exp2', save_mode='grid'):
    os.makedirs(save_dir, exist_ok=True)

    # Read images
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    base_images = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith(supported_extensions)]
    mask_images = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(supported_extensions)]
    
    # check base and mask amount
    if len(mask_images) > len(base_images):
        match_mask_images = random.sample(mask_images, len(base_images))
    elif len(mask_images) < len(base_images):
        miss_count = len(base_images) - len(mask_images)
        extra_masks = random.choices(mask_images, k=miss_count)
        match_mask_images = mask_images + extra_masks
    else:
        match_mask_images = mask_images

    # check device
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    idx = 0
    for i in range(0, len(base_images), batch_size):
        base_batch = base_images[i:i+batch_size]
        mask_batch_paths = match_mask_images[i:i+batch_size]

        base_batch = [Image.open(p).convert("RGB") for p in base_batch]

        # Set mask batch and apply random transform
        mask_batch = []
        for mask_path in mask_batch_paths:
            mask_image = cv2.imread(mask_path)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            _, mask_image = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            erode_mask = cv2.erode(mask_image, kernel, iterations=1)

            mask_pil = Image.fromarray(random_transform(erode_mask))
            mask_batch.append(mask_pil)

        # Set prompt batch
        prompt_batch = set_prompts(batch_size=len(base_batch), data=data_name)

        # Inpaint
        inpainted_images = predict_batch_lora(lora_model_path, base_batch, mask_batch, prompt_batch, device)

        if save_mode == 'grid':
            idx += 1
            combined = combine_image_grid([base_batch, 
                                           mask_batch, 
                                           [img.resize((256,256), Image.BILINEAR) for img in inpainted_images]])
            combined.save(os.path.join(save_dir, f"batch_{idx}.png"))
            
        
        else:
            for k, img in enumerate(inpainted_images):
                img_np = np.array(img)
                if np.max(img_np) == 0:
                    print(f"Warning: Inpainted image {k} is completely black!")
                    continue

                idx += 1
                save_path = os.path.join(save_dir, f'inpaint_{idx}.png')
                img.save(save_path)
                print(f'Inpainted result saved in {save_path}')

    print('Finished!')

def combine_image_grid(image_lists):
    rows = len(image_lists)
    cols = max(len(row) for row in image_lists)

    img_width, img_height = image_lists[0][0].size
    combined = Image.new('RGB', (cols * img_width, rows * img_height))

    for row_idx, row in enumerate(image_lists):
        for col_idx, img in enumerate(row):
            combined.paste(img, (col_idx * img_width, row_idx * img_height))

    return combined

if __name__ == "__main__":
    ## Train
    # train_SD_inpaint_lora(data_dir = r'datasets\AOI__dry_films(blue)_prompt_exp2',
    #     project='exp2_lora_v2',
    #     num_epochs=500,
    #     batch_size=1,
    #     lr=1e-6 # 1e-6 # 5e-7
    #     )
    
    ## Predict
    inpaint_all_lora(base_dir=r'datasets\test\clean_images(689)', # 256x256
                mask_dir=r'datasets\test\masks', # 256x256
                save_dir=r'output\exp2_lora_v2\model_epoch_500',
                lora_model_path=r'checkpoints\exp2_lora_v2\model_epoch_500',
                batch_size=16,
                data_name='exp2', # for setting prompts
                save_mode='single'
                )

