import numpy as np
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader.datasets import InpaintingDataset
from utils.augmentation import ImageOnlyTransform
    
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
