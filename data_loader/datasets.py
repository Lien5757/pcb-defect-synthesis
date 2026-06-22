from torch.utils.data import Dataset
import os
import re
from PIL import Image
from torchvision import transforms
import torch

class PCBInpaintingDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.base_dir = base_dir
        self.image_root = os.path.join(base_dir, "images")
        self.mask_root = os.path.join(base_dir, "masks")
        self.text_root = os.path.join(base_dir, "texts")
        self.transform = transform

        self.image_files = []
        self.class_labels = []
        self.class_to_idx = {}  # Map string → int
        self.idx_to_class = {}  # Map int → string

        class_names = sorted(os.listdir(self.image_root))
        for i, name in enumerate(class_names):
            self.class_to_idx[name] = i
            self.idx_to_class[i] = name

        for class_name in class_names:
            class_id = self.class_to_idx[class_name]
            image_dir = os.path.join(self.image_root, class_name)
            mask_dir = os.path.join(self.mask_root, class_name)
            text_dir = os.path.join(self.text_root, class_name)

            for f in os.listdir(image_dir):
                match = re.search(r"defect_pcb_(\d+)\.png", f)
                if match:
                    index_str = match.group(1)
                    image_path = os.path.join(image_dir, f)
                    mask_path = os.path.join(mask_dir, f"mask_{index_str}.png")
                    text_path = os.path.join(text_dir, f"defect_{index_str}.txt")

                    if os.path.exists(mask_path) and os.path.exists(text_path):
                        self.image_files.append((class_id, index_str, image_path, mask_path, text_path))
                        self.class_labels.append(class_id)

        # Transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        class_id, index_str, img_path, mask_path, text_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda p: 255 if p > 10 else 0)
        with open(text_path, "r") as f:
            prompt = f.read().strip()

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        masked = image * (mask < 0.5).float()

        sample = {
            "defect_image": image,
            "mask": mask,
            "masked_image": masked,
            "prompt": prompt,
            "class_id": class_id,  # Integer class
            "class_name": self.idx_to_class[class_id]  # String name (optional)
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

class InpaintingDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.image_root = os.path.join(base_dir, "images")
        self.mask_root = os.path.join(base_dir, "masks")
        self.text_root = os.path.join(base_dir, "texts")
        self.transform = transform

        self.sample_list = []
        self.class_labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        class_names = sorted(os.listdir(self.image_root))
        for class_id, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = class_id
            self.idx_to_class[class_id] = class_name

            image_class_dir = os.path.join(self.image_root, class_name)
            mask_class_dir = os.path.join(self.mask_root, class_name)
            text_class_dir = os.path.join(self.text_root, class_name)

            for fname in os.listdir(image_class_dir):
                name, ext = os.path.splitext(fname)
                if ext.lower() not in [".png", ".jpg", ".jpeg"]:
                    continue

                image_path = os.path.join(image_class_dir, fname)
                mask_path = os.path.join(mask_class_dir, name + ".png")
                text_path = os.path.join(text_class_dir, name + ".txt")

                if os.path.exists(mask_path) and os.path.exists(text_path):
                    self.sample_list.append((
                        class_id, class_name, name, image_path, mask_path, text_path
                    ))
                    self.class_labels.append(class_id)  # for weighted sampler

        # 預設 transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        class_id, class_name, name, image_path, mask_path, text_path = self.sample_list[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda p: 255 if p > 10 else 0)

        with open(text_path, "r") as f:
            prompt = f.read().strip()

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        ## Masked region : gray
        mask_bin = (mask > 0.5).float() # 1:mask region, 0:preserve
        gray = torch.tensor(0.0).to(image.device)
        masked = image * (1 - mask_bin) + gray * mask_bin

        ## Masked region : black
        # masked = image * (mask < 0.5).float() 
        sample = {
            "defect_image": image,
            "mask": mask,
            "masked_image": masked,
            "prompt": prompt,
            "class_id": class_id,
            "class_name": class_name,
            "file_name": name
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample