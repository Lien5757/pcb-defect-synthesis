# Training Configuration Guide

This guide explains how to configure the SD Inpainting training using JSON config files.

## Quick Start

### Option 1: Use Default Config
```bash
python SD_inpainting_train.py --data_dir ./datasets/train/exp1
```

### Option 2: Use JSON Config File
```bash
python SD_inpainting_train.py --config config/exp1.json
```

### Option 3: Mix Config File + CLI Overrides
```bash
python SD_inpainting_train.py --config config/exp1.json --num_epochs 1000 --batch_size 8
```

## Config File Format

### Minimal Config (data_dir required)
```json
{
  "data_dir": "./datasets/train/exp1"
}
```

### Full Config with All Parameters
```json
{
  "data_dir": "./datasets/train/exp1",
  "project_name": "exp1",
  "is_transform": true,
  "num_epochs": 500,
  "batch_size": 4,
  "lr": 1e-6,
  "weight_decay": 1e-6,
  "use_warmup": true,
  "warmup_ratio": 0.05,
  "use_weighted_sampler": true,
  "min_delta": 0.0001,
  "save_interval": 200,
  "plot_interval": 10
}
```

## Parameter Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | (required) | Path to training data directory |
| `project_name` | str | `"exp1"` | Experiment name (used for output directories) |
| `is_transform` | bool | `false` | Enable data augmentation (crop, flip, resize) |
| `num_epochs` | int | `500` | Number of training epochs |
| `batch_size` | int | `1` | Batch size for training |
| `lr` | float | `5e-7` | Learning rate |
| `weight_decay` | float | `1e-6` | Weight decay for optimizer |
| `use_warmup` | bool | `true` | Enable learning rate warmup |
| `warmup_ratio` | float | `0.05` | Warmup ratio (% of total steps) |
| `use_weighted_sampler` | bool | `true` | Use weighted sampling for class balancing |
| `min_delta` | float | `1e-4` | Minimum loss change for improvement tracking |
| `save_interval` | int | `200` | Save checkpoint every N epochs |
| `plot_interval` | int | `10` | Update plot every N epochs |

## Usage Examples

### Example 1: Quick Experiment
```bash
python SD_inpainting_train.py --config config/exp1.json
```

### Example 2: Override Epochs and Batch Size
```bash
python SD_inpainting_train.py --config config/exp1.json --num_epochs 1000 --batch_size 8
```

### Example 3: Enable Augmentation
```bash
python SD_inpainting_train.py \
  --data_dir ./datasets/train/exp1 \
  --project_name exp1_with_aug \
  --is_transform true \
  --batch_size 4
```

### Example 4: Custom Learning Rate Schedule
```bash
python SD_inpainting_train.py \
  --config config/exp1.json \
  --lr 1e-6 \
  --warmup_ratio 0.1
```

## Creating Your Own Config

1. Copy the example file:
```bash
cp config/exp1_example.json config/my_experiment.json
```

2. Edit the parameters:
```bash
nano config/my_experiment.json
```

3. Run training:
```bash
python SD_inpainting_train.py --config config/my_experiment.json
```

## Error Handling

### Missing data_dir
```
Configuration Error: Config file must contain 'data_dir' parameter.
```
**Solution:** Add `"data_dir"` to your JSON config file.

### Invalid JSON Syntax
```
Configuration Error: Config file is not valid JSON
```
**Solution:** Validate your JSON file using an online JSON validator.

### File Not Found
```
Configuration Error: Config file not found: path/to/config.json
```
**Solution:** Check that the file path is correct and the file exists.

### Invalid Parameter Values
```
Configuration Error: Invalid parameters in config file
```
**Solution:** Check parameter types (int, float, bool, str) match the expected types.

## CLI Arguments Override Config File

When both config file and CLI arguments are provided, **CLI arguments take precedence**:

```bash
# Config file has batch_size=4
python SD_inpainting_train.py --config config/exp1.json --batch_size 8
# Result: batch_size will be 8
```

## Tips

- **Data Augmentation:** Use `is_transform: true` for small datasets (<1000 images)
- **Batch Size:** Larger batches (8-16) speed up training but require more VRAM
- **Learning Rate:** Lower LR (1e-7) for fine-tuning, higher LR (1e-5) for scratch
- **Plot Interval:** Set to 1 for frequent visualization, 50+ for faster training
- **Save Interval:** Checkpoints are saved every N epochs; set larger value to save disk space

## Supported Config Locations

```
project_root/
├── config/
│   ├── exp1_example.json     # Example config
│   ├── my_experiment.json    # Your config
│   └── prompts.json          # Prompt configuration (auto-loaded)
└── SD_inpainting_train.py
```

Config files can be located anywhere; use absolute or relative paths:
```bash
python SD_inpainting_train.py --config ./my_configs/exp1.json
python SD_inpainting_train.py --config /home/user/configs/exp1.json
```
