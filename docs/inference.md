# Inference Guide

Generate synthetic defects using a trained checkpoint.

## Quick Start

**Option 1: Using Dataset Prompts (Recommended)**
```python
from config.enums import PromptMode
from SD_inpainting_predict import Inpainter

inpainter = Inpainter(
    model_path=r"checkpoints\exp1\best_model.pt",
    data_name="exp1",                           # Load prompts from config/prompts.json
    enable_aug_on_base=True,
    enable_aug_on_mask=True
)

inpainter.run(
    base_dir=r"datasets\test\base",             # Directory of clean PCB images
    mask_dir=r"datasets\test\masks",            # Directory of defect masks
    prompt_mode=PromptMode.MULTI,               # Auto-load from config/prompts.json
    batch_size=18
)
```

**Option 2: Using Custom Prompts**
```python
inpainter.run(
    base_dir=r"datasets\test\base",
    mask_dir=r"datasets\test\masks",
    prompt_mode=PromptMode.SINGLE,
    prompt="A dark blue dry film residual defect",
    batch_size=18
)
```

## Inpainter Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model_path` | — | Path to trained checkpoint (e.g., `checkpoints/exp1/best_model.pt`) |
| `data_name` | `"exp1"` | Dataset name for prompt selection (loads from `config/prompts.json`) |
| `enable_aug_on_base` | `True` | Apply augmentation (flip, rotate) to base images |
| `enable_aug_on_mask` | `True` | Apply matching augmentation to masks |
| `scheduler_type` | `"DDIM"` | Noise scheduler type |
| `device` | Auto-detect | Device to use (`"cuda:0"` or `"cpu"`) |

## Run Parameters

| Parameter | Default | Notes |
|---|---|---|
| `base_dir` | — | Directory of clean PCB images (512×512) |
| `mask_dir` | — | Directory of defect masks (512×512 binary) |
| `prompt_mode` | `PromptMode.MULTI` | `PromptMode.MULTI` for auto-loaded prompts or `PromptMode.SINGLE` for custom prompt |
| `prompt` | — | Custom prompt text (required if `prompt_mode=PromptMode.SINGLE`) |
| `batch_size` | `4` | Images per batch (~18 for VRAM-efficient processing) |
| `target_total` | `None` | Total images to generate (replicates batches if needed) |

## Outputs

Generated synthetic defect images are saved to:

```
output/{project_name}/{model_type}_{timestamp}/
├── inpaint_dark_blue_001_0_timestamp.png      # Results with class names
├── inpaint_dark_blue_001_1_timestamp.png
├── inpaint_light_blue_002_0_timestamp.png
└── ...
```

**Output Format:**
- Organized by detected class name from `config/prompts.json`
- Filename format: `inpaint_{class_name}_{batch}_{index}_{timestamp}.png`
- Sequential numbering for easy batch tracking

## Input Format

**Base Images** (`base_dir/`):
- Defect-free PCB photos (512×512 PNG/JPG)
- Clean surfaces where defects will be synthesized
- Example: `datasets/test/base/pcb_001.png`

**Masks** (`mask_dir/`):
- Binary masks (512×512, grayscale PNG)
- White (255) = region to synthesize defects
- Black (0) = preserve original
- Example: `datasets/test/masks/pcb_001.png`

**Prompts** (`config/prompts.json`):
- Auto-loaded based on `data_name` parameter
- Example: `"A dark blue dry film residual defect"`

## Augmentation Options

Inference supports real-time augmentation to generate more diverse synthetic defects:

```python
inpainter = Inpainter(
    model_path=r"checkpoints\exp1\best_model.pt",
    data_name="exp1",
    enable_aug_on_base=True,    # Random flip/rotate base images
    enable_aug_on_mask=True     # Apply matching augmentation to masks (flip, rotate, scale, etc.)
)
```

**Purpose:**
- **Base images:** Random flip / rotate for varied composition
- **Masks:** Random flip / rotate / scale to create diverse defect positions and sizes
- **Result:** Each batch generates different augmented versions for richer synthetic dataset
- No impact on inference quality; purely for data diversity

## Troubleshooting

### Generation Failures

**Semantic Consistency:** Mask shape must match prompt intent.

- ✓ Thin mask + "thin line scratch" → Works
- ✗ Thin mask + "large blob defect" → Generation failure (semantic mismatch)

**If generation fails, check:**
1. Does mask shape match the prompt semantically?
2. Is prompt specific and clear?
3. Was the model trained on similar prompts?
4. Is the mask binary (only 0 and 255)?

### Batch Size

- **Large batches** (18–32): Faster but more VRAM (~11.6 GB)
- **Small batches** (4–8): Slower but memory-efficient
- Adjust based on your GPU memory

---

## Next Steps

1. **[CNN Classifier Gradio Toolkit](https://github.com/Lien5757/cnn-classifier-gradio)** — Mix synthetic defects into classifier training
2. **[Training Guide](training.md)** — Fine-tune on your custom data
3. **[Data Preparation](data_preparation.md)** — Prepare new datasets
