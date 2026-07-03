# Data Preparation Guide

Detailed instructions for preparing images, masks, and prompts.

---

## Step 1: Resize Images to 512×512

Standardize all images to the same dimensions.

**File:** `data_preprocess/pre_image_utils.py`

**Setup:**
```python
if __name__ == "__main__":
    TASK = "resize"  # or "rename"
    
    if TASK == "resize":
        data_dir = r'your/dataset/path'
        resize_images(data_dir, size=(512, 512))
```

**Run:**
```bash
python data_preprocess/pre_image_utils.py
```

**Options:**
- `"resize"` — Resize in-place, keep original filenames
- `"rename"` — Resize + rename by class (`scratch_pcb_defect_001.png`)

**Output:** `images/<class>/` contains resized 512×512 images

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

## Next

See [Dataset Design Guidelines](dataset_design.md) for best practices and troubleshooting.
