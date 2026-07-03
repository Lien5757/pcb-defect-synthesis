# Training Guide

Fine-tune the SD Inpainting UNet on your PCB defect data.

## Quick Start

**Option 1: Using Config File (Recommended)**
```bash
python SD_inpainting_train.py --config config/exp1_example.json
```

**Option 2: Using CLI Parameters**
```bash
python SD_inpainting_train.py \
  --data_dir datasets/train/exp1 \
  --project_name exp1 \
  --num_epochs 500 \
  --batch_size 4 \
  --lr 1e-6 \
  --use_weighted_sampler
```

**Option 3: Config File + CLI Override**
```bash
# Config file defines base settings, CLI args override specific values
python SD_inpainting_train.py --config config/exp1_example.json --num_epochs 300
```

## Configuration File

All hyperparameters are defined in JSON config files (e.g., `config/exp1_example.json`):

```json
{
  "data_dir": "datasets/train/exp1",
  "project_name": "exp1",
  "is_transform": true,
  "num_epochs": 500,
  "batch_size": 4,
  "lr": 1e-06,
  "weight_decay": 1e-06,
  "use_warmup": true,
  "warmup_ratio": 0.05,
  "use_weighted_sampler": false,
  "min_delta": 0.0001,
  "save_interval": 200,
  "plot_interval": 10
}
```

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| `data_dir` | — | Dataset path (must have `images/`, `masks/`, `texts/`) |
| `project_name` | `"exp1"` | Experiment name; auto-generates run ID (e.g., `exp1_base_20260703_10-30-45`) |
| `is_transform` | `false` | Enable data augmentation (flip, rotate) during training |
| `num_epochs` | `500` | Total training epochs |
| `batch_size` | `1` | Limited by VRAM (~20.6 GB for BS=1, ~22.6 GB for BS=4 on RTX 4090) |
| `lr` | `5e-7` | Learning rate for stable fine-tuning |
| `weight_decay` | `1e-6` | L2 regularization |
| `use_warmup` | `true` | Linear warmup schedule for first `warmup_ratio` steps |
| `warmup_ratio` | `0.05` | Warmup as fraction of total steps |
| `use_weighted_sampler` | `false` | Upsample minority classes (critical for imbalanced data) |
| `min_delta` | `1e-4` | Minimum loss improvement threshold for tracking |
| `save_interval` | `200` | Save checkpoint every N epochs |
| `plot_interval` | `10` | Update loss plot every N epochs |


## Auto-Generated Metadata

After training completes, checkpoint directory contains:

```
checkpoints/exp1_base_20260703_10-30-45/
├── config.json          # Exact config used for this run
├── metadata.json        # Training summary (loss, epochs, run_id)
├── best_model.pt        # Best checkpoint (lowest loss)
├── final_model.pt       # Final checkpoint after all epochs
├── latest.pt            # Latest checkpoint (for resuming)
└── training.log         # Full training log
```

**Benefits:**
- `config.json` ensures full reproducibility
- `metadata.json` logs final results for comparison
- Auto-generated run ID (e.g., `exp1_base`, `exp1_transform_lr5e7`) encodes parameter changes

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
