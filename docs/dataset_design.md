# Dataset Design Guidelines

Best practices for structuring defect classes and avoiding common pitfalls.

---

## ✓ Best Practices

### Split Classes by Visual Characteristics

Separate defects by **color** and **shape** to ensure clear, distinct classes:

```
Good Class Design:
├── 0_dark_blue
├── 1_light_blue
├── 2_large_scratch
├── 3_thin_scratch

Poor Class Design (avoid):
├── blue_stuff
├── scratches_and_stuff
├── everything_else
```

- Separate by color: "A dark blue defect" vs. "A light blue defect"
- Separate by shape: "A large area scratch" vs. "A thin line scratch"
- Each class should have **visually consistent** samples

### Use Specific, Descriptive Prompts

Semantic clarity directly affects generation quality:

| Prompt | Quality | Reason |
|--------|---------|--------|
| "A dark blue dry film residual defect" | ✓ Good | Specific color, context, defect type |
| "A large scratch covering PCB" | ✓ Good | Clear visual intent |
| "Blue defect" | ⚠️ Weak | Vague, missing context |
| "Defect" | ✗ Poor | Too generic, unstable generation |
| "Blue and scratchy thing" | ✗ Poor | Compound concepts confuse model |

### Aim for Balanced, Sufficient Sample Counts

- **Target:** 50+ samples per class
- **Minimum:** 30 samples per class for stable generation
- **Distribution:** Roughly equal across classes (use weighted sampling if imbalanced)

### Use Natural, Rough Masks

Masks do not need to be perfectly traced:

```
Good Mask:
├── Freehand, natural edge variation
├── Some fuzzy/rough boundaries
└── Reflects real defect uncertainty

Poor Mask:
├── Over-smoothed, unrealistic edges
├── Perfect pixel-level precision
└── Does not match natural defects
```

- Freehand annotation is **sufficient**
- Natural variation helps model learn diverse shapes
- Fuzzy edges are **acceptable and beneficial**
- Do not over-trace or over-smooth

---

## ✗ Pitfalls to Avoid

### 1. Mixing Visually Diverse Defects in One Class

**Problem:** Combining structurally different defects causes **generation collapse**

```
❌ BAD: "scratch_defect" class contains both:
   ├── Thin line scratches (1-2 pixels wide)
   └── Large area scratches (50+ pixels wide)
   → Result: Color distortion, texture degradation, unstable generation

✓ GOOD: Split into two classes:
   ├── "thin_scratch" (line-like, 1-2 pixels)
   └── "large_scratch" (area-like, 50+ pixels)
   → Result: Stable, controllable generation
```

**Fix:** Define classes by **visual homogeneity**, not defect type alone.

### 2. Too Many Classes with Few Samples

**Problem:** Minority classes produce **severely distorted outputs**

```
❌ BAD: 20 classes, 5 samples each
   → High variance, color confusion, poor generation

✓ GOOD: 5 classes, 50+ samples each
   → Stable generation, clear semantic boundaries
```

**Fix:** 
- Consolidate rare classes
- Collect more samples for minority classes
- Use weighted sampling during training (if imbalance is unavoidable)

### 3. Semantic Mismatch Between Mask and Prompt

**Problem:** Mask shape must match prompt intent

```
❌ MISMATCH:
   Prompt: "A large blob defect covering wide area"
   Mask: Thin line (1 pixel wide)
   → Generation failure (semantic contradiction)

✓ CORRECT:
   Prompt: "A thin line scratch"
   Mask: Thin line (1-2 pixels wide)
   → Stable generation
```

**This is NOT a model bug—it's a data consistency issue.**

**Fix:** Review all prompts and masks together. Ensure:
- Thin masks pair with "thin" prompts
- Large masks pair with "large" or "area" prompts
- Color in prompts matches visual characteristics of mask regions

### 4. Overly Complex or Vague Prompts

**Problem:** Stable Diffusion is sensitive to prompt clarity

```
❌ Poor:
   "There might be a blue defect or maybe a scratch"
   "Defect-like thing on the board"
   "Possibly some discoloration"

✓ Good:
   "A dark blue dry film residual defect"
   "A thin scratching defect on PCB"
   "A white contamination mark"
```

**Fix:** Use **concrete, visual descriptors** only. Avoid hedging language.

---

## Troubleshooting

### Masks Not Being Drawn

**Symptom:** Mouse input not registering in mask annotation tool

**Checklist:**
- [ ] `masks/` directory exists and is writable
- [ ] File permissions correct (not read-only)
- [ ] Image format supported (PNG/JPG)
- [ ] Try simpler image file first to isolate issue

---

### Some Images Missing Masks or Prompts

**Symptom:** `final_check.py` reports missing files

```
Found N missing files:
├── image_001.png (missing mask)
├── image_002.png (missing prompt)
└── ...
```

**Fix:**
```bash
# Identify exact missing files
python data_preprocess/final_check.py

# Return to Step 2 to complete mask annotation
python data_preprocess/pre_mask_utils.py

# Return to Step 3 to generate prompts
python data_preprocess/pre_prompt_utils.py

# Verify again
python data_preprocess/final_check.py
```

---

### Generation Later Fails with Semantic Mismatch

**Symptom:** Model generates distorted/incorrect outputs during inference

**Investigation:**
```bash
# Review prompts
ls texts/<class>/

# Review masks
ls masks/<class>/

# Are they semantically aligned?
# Example: Do all "thin_scratch" prompts have thin masks?
```

**Fix Example:**

If generating class `large_scratch` but outputs are distorted:
1. Check prompts in `texts/large_scratch/` — are they explicit about size?
2. Check masks in `masks/large_scratch/` — do they actually cover large areas?
3. If mismatch: Redefine class boundary or redraw masks

---

### Text Generation Creates Empty Files

**Symptom:** Prompt files exist but are blank

**Causes:**
- `CLASS_TO_PROMPT` dict missing a class name from `images/`
- Class name mismatch (case-sensitive!)

**Fix:**
```python
# Verify all classes in images/ are in CLASS_TO_PROMPT
import os
image_classes = os.listdir('data/images')
print(image_classes)

# Add missing classes to CLASS_TO_PROMPT
CLASS_TO_PROMPT = {
    "class_1": "...",
    "class_2": "...",
    # ← Ensure all image classes are here
}
```

---

## Data Statistics Checklist

Before proceeding to training, verify all items:

- [ ] All images resized to 512×512
- [ ] All images have corresponding binary masks
- [ ] All images have corresponding prompt files
- [ ] `final_check.py` outputs "All data is properly paired!" ✓
- [ ] Each class has ≥ 30 samples (50+ recommended)
- [ ] Class distribution is roughly balanced
- [ ] Prompts are specific and descriptive (no hedging language)
- [ ] Mask shapes are semantically consistent with prompts
- [ ] No classes mixing visually diverse defects
- [ ] No extreme class imbalance (if unavoidable, plan weighted sampling)

---

## Next Steps

Once all checks pass:
1. **[Training Guide](training.md)** — Fine-tune SD Inpainting on your data
2. **[Inference Guide](inference.md)** — Generate synthetic defects
3. **[CNN Classifier Toolkit](https://github.com/Lien5757/cnn-classifier-gradio)** — Train classifier on mixed synthetic+real data
