import numpy as np
from collections import Counter
from typing import Tuple, List
from torch.utils.data import DataLoader, WeightedRandomSampler
from data_loader.datasets import InpaintingDataset
from utils.augmentation import ImageOnlyTransform
    
def load_train_data(
    data_dir: str,
    batch_size: int,
    isTransform: bool = False,
    use_weighted_sampler: bool = True
) -> Tuple[InpaintingDataset, DataLoader, List[float]]:
    """Load training dataset with optional augmentation and weighted sampling.

    Args:
        data_dir: Path to dataset root directory.
        batch_size: Batch size for DataLoader.
        isTransform: Enable data augmentation (crop, flip, resize).
        use_weighted_sampler: Use weighted sampling to balance class distribution.

    Returns:
        Tuple of (dataset, dataloader, sample_weights).
    """
    if isTransform:
        transform = ImageOnlyTransform(crop_size=(448, 448), flip_prob=0.5)
    else:
        transform = None
    dataset = InpaintingDataset(base_dir=data_dir, transform=transform)

    if use_weighted_sampler:
        sample_weights = compute_soft_weights(dataset.class_labels, max_clip=0.05)
    else:
        sample_weights = [1.0] * len(dataset)

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)
    return dataset, dataloader, sample_weights

def compute_soft_weights(group_labels: List[int], max_clip: float = 0.05) -> List[float]:
    """Compute soft class weights with clipping and normalization.

    Args:
        group_labels: List of class labels for each sample.
        max_clip: Maximum weight value after clipping.

    Returns:
        Normalized sample weights.

    Raises:
        ValueError: If computed weights sum to zero.
    """
    group_counts = Counter(group_labels)
    print("class counts:", group_counts)

    weights = {cls: 1.0 / (count + 1e-6) for cls, count in group_counts.items()}
    weights = {cls: min(w, max_clip) for cls, w in weights.items()}

    total = sum(weights.values())
    weights = {cls: w / total for cls, w in weights.items()}

    sample_weights = [weights[label] for label in group_labels]

    if sum(sample_weights) == 0:
        raise ValueError("Sample weights sum to 0. Check class distribution or clipping value.")

    return sample_weights
