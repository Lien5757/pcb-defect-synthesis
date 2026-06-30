# Data Preparation Guide

Prepare PCB defect data for SD Inpainting training. Follow these 4 steps:

```
data/
├── images/           # Defect images by class
├── masks/            # Binary masks (white = defect region)
└── texts/            # Text prompts per class
```

---

## Quick Start (5 min)

```bash
# 1. Resize images to 512×512
python data_preprocess/pre_image_utils.py

# 2. Annotate masks (interactive)
python data_preprocess/pre_mask_utils.py

# 3. Generate text prompts
python data_preprocess/pre_prompt_utils.py

# 4. Verify data is complete
python data_preprocess/final_check.py
```

---

## Detailed Setup

- **[Step 1: Image Resizing](data_annotation.md#step-1-resize-images)**
- **[Step 2: Mask Annotation](data_annotation.md#step-2-annotate-masks)**
- **[Step 3: Prompt Generation](data_annotation.md#step-3-generate-prompts)**
- **[Step 4: Verification](data_annotation.md#step-4-verify-data)**

## Best Practices

- **[Dataset Design Guidelines](dataset_design.md)**
- **[Troubleshooting](dataset_design.md#troubleshooting)**

---

## Requirements Checklist

Before training:

- [ ] All images 512×512
- [ ] Every image has a binary mask
- [ ] Every image has a text prompt
- [ ] `final_check.py` passes ✓
- [ ] ≥ 30 samples per class (50+ recommended)
- [ ] Roughly balanced class distribution

---

**Next:** [Training Guide](training.md) or [CNN Classifier Toolkit](https://github.com/Lien5757/cnn-classifier-gradio)
