# PCB Defect Synthesis via Stable Diffusion Inpainting

Fine-tuned Stable Diffusion Inpainting for industrial PCB defect data augmentation —  
generating realistic, localized defects from clean images, binary masks, and text prompts,  
validated on a 15-class EfficientNet-B0 classifier.

> M.S. Thesis — National Taipei University of Technology, 2025  
> *PCB Defect Generation & Detection Techniques based on Diffusion Model*

---

## Highlights

- **Stable Diffusion Inpainting fine-tuned with LoRA** on real PCB defect samples
- Controlled synthesis via **clean image + defect mask + text prompt**
- Outperforms GauGAN baseline by ~4% recall at 2x augmentation
- Mask precision and prompt clarity are the dominant factors governing generation quality
- Explored **Weighted Sampling** and **AMP** for low-data industrial training scenarios
- Validated on **15 defect classes** using EfficientNet-B0

---

## Problem

Real-world PCB inspection datasets suffer from two fundamental challenges:

- **Scarcity** — defective samples are rare by nature of the manufacturing process
- **Class imbalance** — some defect types have far fewer samples than others

Standard augmentation (flip, crop, color jitter) cannot synthesize structurally realistic defects.  
This study validates diffusion models as a practical solution for industrial defect data augmentation.

---

## Method — Stable Diffusion Inpainting

A Stable Diffusion Inpainting model is fine-tuned on real PCB defect samples via LoRA.  
At inference, given a **clean image + defect mask + text prompt**, the model synthesizes  
a localized defect in the masked region — enabling controlled, realistic augmentation.

```
Prompt template: "A {class_name} {defect_type} defect"
```

> Key finding: generation quality is highly sensitive to **mask precision** and  
> **prompt semantic clarity** — quality over quantity applies here.

**Why inpainting over other approaches?**  
Unlike patch-level generation (DDPM + DIP) or unconditional synthesis, SD Inpainting  
preserves the surrounding PCB context and enables explicit spatial control via masks,  
which is essential for industrial inspection where defect location and boundary matter.

---

## Experimental Design

**Quantitative:** EfficientNet-B0 trained on a 15-class PCB defect dataset,  
evaluated across augmentation methods and ratios (Precision / Recall / F1).

**Qualitative:** Three datasets of increasing semantic and structural complexity  
were designed to evaluate SD Inpainting behavior under varying conditions:

| Condition Tested | Finding |
|-----------------|---------|
| Mask precision | Fine-grained masks produce significantly better generation quality |
| Prompt clarity | Semantically clear prompts improve generation stability |
| Dataset complexity | Higher structural complexity reduces generation controllability |
| Weighted Sampling | Improves recall on minority defect classes |
| AMP training | Faster training with no significant accuracy loss |
| LoRA fine-tuning | Effectively injects domain knowledge from limited samples |

---

## Results

### SD Inpainting vs. Baselines — Recall @ 2x Augmentation

All methods trained on EfficientNet-B0, 15-class PCB defect classifier.

| Method | Recall |
|--------|--------|
| GauGAN (generative baseline) | 82.9% |
| DDPM + DIP (diffusion baseline) | 86.9% |
| **SD Inpainting (this repo)** | **86.4%** |

SD Inpainting achieves +3.5% over GauGAN while offering superior **spatial controllability**  
and **prompt-guided synthesis** — advantages that raw recall numbers do not fully capture.

### Effect of Augmentation Ratio — SD Inpainting (single minority class)

| Config | Train Size | Recall |
|--------|------------|--------|
| A — No augmentation | ~500 | ~0.78 |
| B — 2x | ~1,050 | ~0.80 |
| C — 3x | ~1,550 | ~0.82 |
| D — 5x | ~3,200 | ~0.90 |

---

## Repository Structure

```
.
├── 0_Introduction/          # Dataset overview and class definitions
├── 1_Preprocess/
│   ├── pre_image_utils.py   # Image renaming and resizing
│   ├── pre_mask_utils.py    # Interactive mask annotation tool
│   ├── pre_prompt_utils.py  # Text prompt generation per defect class
│   └── final_check.py       # Data completeness verification
├── 2_Training/
│   └── SD_inpainting_train.py
├── 3_Prediction/
│   └── SD_inpainting_predict.py
└── requirements.txt
```

---

## Setup

### Environment

```bash
# Python 3.10 + CUDA 11.8 required
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers==4.46.3 peft
pip install opencv-python matplotlib scikit-image tqdm
```

### Hardware Requirements

| Stage | VRAM Usage | Device Tested |
|-------|-----------|---------------|
| Training (AMP, BS=1) | 20.6 / 24 GB | RTX 4090 |
| Training (AMP, BS=4) | 22.6 / 24 GB | RTX 4090 |
| Inference (AMP, BS=1) | 11.6 / 24 GB (~3.4s/img) | RTX 4090 |

> Minimum **24 GB VRAM** required for training.  
> Inference may be possible on 12 GB GPUs.  
> Cloud alternative: Google Colab with A100.

---

## Data Preparation

This repository does not include the PCB dataset. Prepare your own data in the following structure:

```
data/
├── images/          # Defect images, named: pcb_defect_{idx}.png
├── masks/           # Binary masks marking defect regions
└── prompts/         # Text prompts per defect class (auto-generated)
```

**Naming convention:** filenames encode defect type — all files within the same class  
share a consistent prefix (e.g., `scratch_001.png`, `scratch_002.png`).

---

## Usage

### Step 1 — Preprocess

```bash
# 1. Rename and resize images to 512x512
python 1_Preprocess/pre_image_utils.py --data_dir ./data

# 2. Annotate defect masks interactively
python 1_Preprocess/pre_mask_utils.py --data_dir ./data
```

**Mask annotation keyboard shortcuts:**

| Key | Action |
|-----|--------|
| `1` / `2` / `3` | Draw mask (brush sizes) |
| `D` | Delete / erase |
| `B` | Previous image |
| `S` | Save current mask |
| `P` / `N` | Navigate prev / next |

```bash
# 3. Generate text prompts per defect class
python 1_Preprocess/pre_prompt_utils.py --data_dir ./data

# 4. Verify all data is ready
python 1_Preprocess/final_check.py --data_dir ./data
```

### Step 2 — Training

```bash
python 2_Training/SD_inpainting_train.py --data_dir ./data
```

### Step 3 — Inference

```bash
python 3_Prediction/SD_inpainting_predict.py \
  --data_dir ./data \
  --output_dir ./output
```

Output folders generated:

| Folder | Content |
|--------|---------|
| `Inpainted_results/` | Final synthetic defect images |
| `Combine_grid/` | Side-by-side comparison grids |
| `Masks/` | Applied masks |
| `Batch_grid/` | Batch overview grids |

---

## Related Tool

The EfficientNet-B0 classifier used in this study is available as a standalone interactive toolkit:

**[CNN Classifier Gradio Toolkit](https://github.com/Lien5757/cnn-classifier-gradio)**

| Feature | Description |
|---------|-------------|
| Models | ResNet18, EfficientNet-B0, DenseNet121 |
| Training | Real-time loss / accuracy curves |
| Evaluation | Confusion matrix, Precision / Recall / F1 |
| Explainability | Grad-CAM heatmap visualization |
| Interface | Gradio web UI — no coding required |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Generative Models | Stable Diffusion Inpainting (runwayml/stable-diffusion-inpainting) |
| Fine-tuning | LoRA (via `peft`) |
| Classification | EfficientNet-B0, PyTorch |
| Training Techniques | AMP, Weighted Sampling |
| Image Processing | OpenCV |
| UI / Demo | Gradio |

---

## Citation

```bibtex
@mastersthesis{lian2025pcb,
  title  = {PCB Defect Generation and Detection Techniques based on Diffusion Model},
  author = {Lian, Julie},
  school = {National Taipei University of Technology},
  year   = {2025}
}
```
