import random
import numpy as np
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader.datasets import InpaintingDataset

class ImageOnlyTransform:
    def __init__(self, crop_size=(448, 448), flip_prob=0.5):
        self.crop_size = crop_size
        self.flip_prob = flip_prob

    def __call__(self, sample):
        img, mask, masked = sample["defect_image"], sample["mask"], sample["masked_image"]

        # Sample consistent crop
        i, j, h, w = transforms.RandomCrop.get_params(img, self.crop_size)
        img = transforms.functional.crop(img, i, j, h, w)
        mask = transforms.functional.crop(mask, i, j, h, w)
        masked = transforms.functional.crop(masked, i, j, h, w)

        # Flip
        if random.random() < self.flip_prob:
            img = transforms.functional.hflip(img)
            mask = transforms.functional.hflip(mask)
            masked = transforms.functional.hflip(masked)

        # Resize with proper interpolation
        img = transforms.functional.resize(img, [512, 512], interpolation=transforms.InterpolationMode.BILINEAR)
        mask = transforms.functional.resize(mask, [512, 512], interpolation=transforms.InterpolationMode.NEAREST)
        masked = transforms.functional.resize(masked, [512, 512], interpolation=transforms.InterpolationMode.BILINEAR)

        # Optional: Skip sample if mask is too small
        if mask.sum() < 10:
            return None  # Or re-sample

        sample["defect_image"] = img
        sample["mask"] = mask
        sample["masked_image"] = masked
        return sample
    
def load_train_data(data_dir, batch_size, isTransform=False, use_weighted_sampler=True):
    if isTransform:
        transform = ImageOnlyTransform(crop_size=(448, 448), flip_prob=0.5)
    else:
        transform = None
    dataset = InpaintingDataset(base_dir=data_dir, transform=transform)

    ## sample weights
    if use_weighted_sampler:
        sample_weights = compute_soft_weights(dataset.class_labels, max_clip=0.05)
    else:
        sample_weights = [1.0] * len(dataset)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    return dataset, dataloader, sample_weights

def compute_weights(dataset):
    # Count class frequencies
    class_counts = np.bincount(dataset.class_labels)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[c] for c in dataset.class_labels]
    return sample_weights

def compute_soft_weights(group_labels, max_clip=0.05):
    group_counts = Counter(group_labels)
    print("class counts:", group_counts)

    # Step 1: inverse weighting
    weights = {cls: 1.0 / (count + 1e-6) for cls, count in group_counts.items()}

    # Step 2: clip too large weights
    weights = {cls: min(w, max_clip) for cls, w in weights.items()}

    # Step 3: normalize
    total = sum(weights.values())
    weights = {cls: w / total for cls, w in weights.items()}

    # Step 4: map to sample weights
    sample_weights = [weights[label] for label in group_labels]

    # Optional safety check
    if sum(sample_weights) == 0:
        raise ValueError("Sample weights sum to 0. Check class distribution or clipping value.")

    return sample_weights
