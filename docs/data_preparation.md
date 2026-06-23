# Data Preparation Guide

This guide walks through the complete pipeline for preparing PCB defect data for training the SD Inpainting model.

## Dataset Organization

Organize your raw data into the following structure:

```
data/
├── images/           # Defect images (organized by class subfolder)
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── masks/            # Binary masks marking defect regions
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── texts/            # Text prompts per class (auto-generated)
    ├── class_1/
    ├── class_2/
    └── ...
```

**Key format requirements:**
- **Images:** PNG/JPG format, any size initially
- **Masks:** Binary PNG, white (255) = defect region to synthesize, black (0) = preserve as-is
- **Prompts:** Plain text files with semantic description per class

---

## Step-by-Step Preparation

### Step 1: Resize Images to 512×512

Standardize image dimensions across your dataset.

**File:** `data_preprocess/pre_image_utils.py`

**Setup:**
```python
if __name__ == "__main__":
    TASK = "resize"  # or "rename"
    
    if TASK == "resize":
        data_dir = r'your/dataset/path'  # ← Modify this
        resize_images(data_dir, size=(512, 512))
```

**Run:**
```bash
python data_preprocess/pre_image_utils.py
```

**Options:**
- **`TASK = "resize"`** — Resize in-place, keep original filenames
- **`TASK = "rename"`** — Resize + rename sequentially by class (`scratch_pcb_defect_001.png`, etc.)

**Output:** Resized images in `images/<class>/`

---

### Step 2: Annotate Masks Interactively

Draw binary masks marking the defect regions manually.

**File:** `data_preprocess/pre_mask_utils.py`

**Setup:**
```python
if __name__ == "__main__":
    MODE = "draw_all_classes"  # or "draw_single_class"
    
    if MODE == "draw_all_classes":
        data_dir = r'your/dataset/path'  # ← Modify this
        # Will iterate through all classes in images/
        
    elif MODE == "draw_single_class":
        data_dir = r'your/dataset/path'
        defect_type = 'scratch'  # ← Specify target class
```

**Run:**
```bash
python data_preprocess/pre_mask_utils.py
```

**Mouse Controls:**
- **Left-click + drag** — Freehand drawing (white = defect area)
- **'m'** — Toggle between Freehand and Line mode
- **'1', '2', '3'** — Brush size (Small, Medium, Large)
- **'b'** — Undo last stroke
- **'d'** — Discard existing mask, start fresh
- **'s'** — Save and move to next image
- **'n'** — Skip to next image
- **'p'** — Back to previous image
- **'q'** — Quit

**Important:** Freehand annotation is sufficient. Natural edge variation helps the model learn diverse defect shapes. Do not over-smooth or over-precise.

**Output:** Binary masks saved to `masks/<class>/` with same filenames as images

---

### Step 3: Generate Text Prompts

Create semantic descriptions for each defect class.

**File:** `data_preprocess/pre_prompt_utils.py`

**Setup:**
```python
# Define class_name -> prompt mapping at top of file
CLASS_TO_PROMPT = {
    "0_crash": "A crash defect on tray",
    "1_scratch": "A scratch defect on tray",
    "2_white_contamination": "White contamination defect on tray",
    # ... add your classes
}

if __name__ == "__main__":
    data_dir = r'your/dataset/path'  # ← Modify this
    save_prompts(data_dir, CLASS_TO_PROMPT)
```

**Run:**
```bash
python data_preprocess/pre_prompt_utils.py
```

**Prompt Guidelines:**
- Be **specific and descriptive**: "A dark blue dry film residual defect" is better than "A blue defect"
- Include **visual characteristics**: color, shape, material context
- Keep **semantic clarity**: avoid mixing multiple visual concepts in one class
- Use **consistent structure** across similar classes

**Examples:**
```
"A dark blue dry film residual defect"
"A light blue dry film residual defect"
"A large scratch covering wide area on PCB"
"A thin single-line scratch on PCB surface"
```

**Output:** Text files saved to `texts/<class>/` with prompt per image

---

### Step 4: Verify Data Completeness

Check that every image has corresponding mask and prompt files.

**File:** `data_preprocess/final_check.py`

**Setup:**
```python
if __name__ == "__main__":
    data_dir = r'your/dataset/path'  # ← Modify this
    check_data_all_set(data_dir=data_dir)
```

**Run:**
```bash
python data_preprocess/final_check.py
```

**Output:**
- ✓ All data is properly paired! (success)
- ✗ Found N missing files (lists which images lack masks/prompts)

**Action:** If missing files are reported, go back to Steps 2 or 3 to complete annotation.

---

## Dataset Design Guidelines

Based on experimental results across multiple dataset configurations, follow these best practices:

### ✓ Do

**Split classes by visual characteristic**
- Separate by color: "A dark blue dry film defect" vs. "A light blue dry film defect"
- Separate by shape: "A large area scratch" vs. "A thin line scratch"
- Each class should have clear, visually consistent samples

**Use descriptive, specific prompts**
- Good: `"A dark blue dry film residual defect"`
- Poor: `"A blue defect"`, `"defect"`, `"blue thing"`

**Aim for balanced class distribution**
- Target: 50+ samples per class
- Distribution: Roughly equal across classes (if possible)
- Minimum: At least 30 samples per class for stable generation

**Use rough freehand masks**
- Natural edge variation helps the model learn diverse defect shapes
- Do not over-smooth or over-trace
- Fuzzy edges are acceptable and beneficial

### ✗ Avoid

**Mixing visually diverse defects into a single class**
- Causes ambiguous semantics
- Results in generation collapse (color distortion, texture degradation)
- Example: Combining "large area scratch" + "thin line scratch" in one class

**Many classes with very few samples**
- Leads to color confusion between classes
- Minority classes produce severely distorted outputs
- Use weighted sampling (see Training guide) if imbalance is unavoidable

**Semantically inconsistent mask shapes**
- Mask must match the prompt intent
- Example: Thin mask + "large blob defect" prompt = generation failure
- This is a semantic mismatch, not a model issue

**Overly complex or vague prompts**
- Stable Diffusion is sensitive to prompt clarity
- Stick to concrete, visual descriptors

---

## Troubleshooting

### Issue: Masks not being drawn
- Ensure `masks/` directory exists
- Check file permissions for the target directory
- Verify image format (PNG/JPG supported)

### Issue: Some images missing masks
- Run `final_check.py` to identify which images
- Return to Step 2 and complete annotation

### Issue: Text generation creates empty files
- Verify `CLASS_TO_PROMPT` dict has all class names present in `images/`
- Class names must match exactly (case-sensitive)

### Issue: Generation later fails with semantic mismatch
- Review prompts in `texts/<class>/` — are they clear and specific?
- Review masks in `masks/<class>/` — do they match the semantic intent of the prompt?
- Example fix: If generating "large blob" but mask is thin line, redefine class or redraw masks

---

## Data Statistics Checklist

Before proceeding to training, verify:

- [ ] All images resized to 512×512
- [ ] All images have corresponding binary masks
- [ ] All images have corresponding prompt files
- [ ] `final_check.py` reports "All data is properly paired!"
- [ ] Each class has ≥ 30 samples (50+ recommended)
- [ ] Class distribution is relatively balanced
- [ ] Prompts are specific and descriptive
- [ ] Mask shapes are semantically consistent with prompts

---

## Next Steps

Once data preparation is complete, proceed to:
1. **Training** — See main README.md for training the SD Inpainting model
2. **Inference** — Generate synthetic defects using trained checkpoint
3. **Classifier Training** — Mix synthetic defects into classifier training set

See [CNN Classifier Gradio Toolkit](https://github.com/Lien5757/cnn-classifier-gradio) for classifier training workflow.
