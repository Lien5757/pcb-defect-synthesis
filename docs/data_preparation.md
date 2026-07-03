# Data Preparation Guide

Prepare PCB defect data for SD Inpainting training. Follow these 4 steps:

```
datasets/train/exp1
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

## Detailed Instructions

- **[Step 1: Image Resizing](#step-1-resize-images-to-512×512)**
- **[Step 2: Mask Annotation](#step-2-annotate-masks)**
- **[Step 3: Prompt Generation](#step-3-generate-prompts)**
- **[Step 4: Verification](#step-4-verify-data)**

---

## Step 1: Resize Images to 512×512

Standardize all images to the same dimensions.

**File:** `data_preprocess/pre_image_utils.py`

**Setup:**
```python
if __name__ == "__main__":
    # ============ CONFIG ============
    TASK = "resize"  # "resize" or "rename"
    src_dir = r'your/dataset/path'
    dst_dir = r'datasets\train\exp1'
    
    # ============ Task 1: Resize (keep original filenames) ============
    if TASK == "resize":
        resize_images(src_dir=src_dir, dst_dir=dst_dir, size=(512, 512))
    
    # ============ Task 2: Resize + Rename ============
    elif TASK == "rename":
        change_filename_with_class(src_dir=src_dir, dst_dir=dst_dir,
                                   defect_name='defect', size=(512, 512))
```

**Run:**
```bash
python data_preprocess/pre_image_utils.py
```

**Configuration:**
- `src_dir` — Source directory containing class-organized raw images
- `dst_dir` — Destination directory (e.g., `datasets\train\exp1`)
- `TASK` — Choose between `"resize"` or `"rename"`

**Task Options:**
| Task | Behavior | Use When |
|------|----------|----------|
| `"resize"` | Resize only, keep original filenames | You want to preserve original naming |
| `"rename"` | Resize + rename by class with sequential numbering | You need standardized `class_XXX.png` format |

**Output:** `{dst_dir}/images/<class>/` contains resized 512×512 images

---

## Step 2: Annotate Masks

Interactively draw binary masks marking defect regions.

**File:** `data_preprocess/pre_mask_utils.py`

**Setup:**
```python
if __name__ == "__main__":
    MODE = "draw_all_classes"  # or "draw_single_class"
    
    if MODE == "draw_all_classes":
        data_dir = r'your/dataset/path'
        # Processes all classes in images/
```

**Run:**
```bash
python data_preprocess/pre_mask_utils.py
```

**Controls:**
| Key | Action |
|-----|--------|
| Left-click + drag | Freehand draw (white = defect) |
| `m` | Toggle Freehand ↔ Line mode |
| `1`, `2`, `3` | Brush size (Small → Large) |
| `b` | Undo last stroke |
| `d` | Clear mask, start fresh |
| `s` | Save and move to next |
| `n` | Skip to next image |
| `p` | Back to previous |
| `q` | Quit |

**Tip:** Rough freehand masks are fine. Natural edge variation helps the model learn diverse defect shapes—do not over-smooth or over-trace.

**Output:** `masks/<class>/` with same filenames as images

---

## Step 3: Generate Prompts

Create semantic descriptions for each defect class from centralized configuration.

**File:** `data_preprocess/pre_prompt_utils.py`  
**Config:** `config/prompts.json` (defines all prompts by dataset name)

**Setup:**
```python
if __name__ == "__main__":
    data_dir = r'your/dataset/path'
    data_name = 'exp1'  # or 'exp2', 'exp3', 'exp4', 'exp5', 'exp5_v2', 'exp6'
    save_prompts(data_dir, data_name=data_name)
```

The script automatically loads prompts for your `data_name` from `config/prompts.json`.

**Run:**
```bash
python data_preprocess/pre_prompt_utils.py
```

**Prompt Guidelines:**
- **Specific** — "A dark blue dry film defect" not "A blue defect"
- **Visual clarity** — Include color, shape, texture
- **Consistent** — Use similar structure across related classes
- **Concrete** — Avoid vague or compound concepts

**Output:** `texts/<class>/` with one prompt per image file

---

## Step 4: Verify Data

Ensure every image has corresponding mask and prompt.

**File:** `data_preprocess/final_check.py`

**Setup:**
```python
if __name__ == "__main__":
    data_dir = r'your/dataset/path'
    check_data_all_set(data_dir=data_dir)
```

**Run:**
```bash
python data_preprocess/final_check.py
```

**Output:**
- ✓ "All data is properly paired!" → Ready to train
- ✗ "Found N missing files" → Lists which images need attention

**Action:** If files are missing, return to Step 2 or 3 to complete annotation.

---

## Best Practices

- **[Dataset Design Guidelines](dataset_design.md)**

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
