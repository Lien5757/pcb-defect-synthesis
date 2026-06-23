# Training Guide

Fine-tune the SD Inpainting UNet on your PCB defect data:

```bash
python SD_inpainting_train.py \
  --data_dir ./data \
  --project_name exp1 \
  --num_epochs 500 \
  --batch_size 1 \
  --lr 5e-7 \
  --weight_decay 1e-6 \
  --warmup_ratio 0.05
```

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--data_dir` | — | Dataset path (must have `images/`, `masks/`, `texts/`) |
| `--project_name` | — | Experiment name; checkpoints → `checkpoints/{project_name}/` |
| `--num_epochs` | 500 | Early stopping applied based on validation loss |
| `--batch_size` | 1 | Limited by VRAM (~20.6 GB for BS=1, ~22.6 GB for BS=4 on RTX 4090) |
| `--lr` | 5e-7 | Learning rate for stable fine-tuning |
| `--weight_decay` | 1e-6 | L2 regularization |
| `--warmup_ratio` | 0.05 | Warmup as fraction of total steps |
| `--use_weighted_sampler` | False | Upsample minority classes (critical for imbalanced data) |

## For Imbalanced Datasets

Add `--use_weighted_sampler` to handle severe class imbalance:

```bash
python SD_inpainting_train.py \
  --data_dir ./data \
  --project_name exp1 \
  --use_weighted_sampler
```

## GPU Memory

- **Training:** 20–23 GB depending on batch size
- **Inference:** ~11.6 GB

## Key Points

- **Quality over quantity:** 50 well-annotated samples beat 500 vague ones
- **Semantic clarity:** Vague semantics cause generation collapse (color/texture distortion)
- **Weighted sampling:** Essential for imbalanced data (prevents minority class distortion)
- **Semantic consistency:** Mask shape must match prompt intent

---

Next: [Inference Guide](inference.md) for generating synthetic defects.  
Setup: [Data Preparation Guide](data_preparation.md)
