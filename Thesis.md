# PCB Defect Generation via Stable Diffusion Inpainting

> **TL;DR** — Not enough defect samples for training? Use a diffusion model to synthesize realistic defects and mix them into your training set. In our experiments, recall on a rare defect class improved from **73% → 92%**.

This project addresses a core challenge in industrial defect detection: **scarce defect samples and severe class imbalance**. We propose a diffusion-based data augmentation pipeline and validate it on real PCB inspection data.

Based on the master's thesis: *PCB Defect Generation and Detection Techniques Based on Diffusion Models*, National Taipei University of Technology, 2025.

---

## Why This Approach?

In real-world PCB production:

- Good-quality images number in the thousands; defect images may be only dozens
- Defect categories are severely imbalanced across the dataset
- Traditional augmentation (rotation, flip) only applies geometric transforms — it cannot add semantic diversity
- Training deep learning models directly on imbalanced data leads to poor recall on rare classes (baseline in this work: **73%**)

**The core idea is straightforward:** train a generative model that can "paint" defects, then have it synthesize realistic defects onto clean PCB images in specified regions — and add those to your training set.

---

## Method Overview

This repo implements two complementary generation strategies, with **Method 2 (SD Inpainting)** as the primary focus:

| | Method 1: DDPM + DIP | Method 2: SD Inpainting ★ |
|---|---|---|
| Core idea | Generate defect region → embed into clean background via image processing | Directly generate defect inside a masked region of a normal image |
| Mask annotation | Requires precise annotation (e.g. LabelMe) | Rough freehand brush is sufficient |
| Semantic control | None (purely visual) | Yes — guided by text prompt |
| Recall @ 2× augmentation | 86.9% | 86.4% |
| Best for | High positional precision | Multi-class semantic control |

---

## Getting Started

### Requirements

**Python 3.10 + CUDA 11.8** required.

Install PyTorch first:
```bash
pip install torch==2.4.1 torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install remaining dependencies:
```bash
pip install -r requirements.txt
```

Includes: `diffusers`, `transformers==4.46.3`, `peft`, `opencv-python`, `matplotlib`, `scikit-image`, `tqdm`.

### Data Preparation

Organize your data as follows:

```
data/
├── images/           # Defect images (all classes mixed)
├── masks/            # Binary masks marking defect regions (white = defect area)
└── prompts/          # Text prompts per defect class (auto-generated)
```

**Data preparation steps:**

```bash
# 1. Resize images to 512×512
python data_preprocess/pre_image_utils.py --data_dir ./data

# 2. Annotate masks interactively (draw defect regions)
python data_preprocess/pre_mask_utils.py --data_dir ./data

# 3. Generate text prompts per class
python data_preprocess/pre_prompt_utils.py --data_dir ./data

# 4. Verify data completeness
python data_preprocess/final_check.py --data_dir ./data
```

**Mask format:** Binary image matching source resolution. **White = region to synthesize defect into; Black = preserve as-is.** Freehand annotation is sufficient.

---

## Training the SD Inpainting Model

### Recommended: UNet Fine-tuning

Fine-tune the full UNet component on your PCB defect samples:

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

**Key parameters:**
- `--lr 5e-7`: Low learning rate for stable fine-tuning
- `--warmup_ratio 0.05`: Gradual warmup for first 5% of training
- `--batch_size 1`: Limited by VRAM (typical on 24GB GPUs); increase if possible
- `--num_epochs 500`: Early stopping applied internally based on validation loss

**GPU Requirements:**
- Training: ~20.6 GB VRAM (RTX 4090, BS=1) or ~22.6 GB (BS=4)
- Inference: ~11.6 GB VRAM (RTX 4090)

Checkpoints are saved to `checkpoints/{project_name}/`.

### Alternative: LoRA Fine-tuning

For memory-constrained setups, use LoRA-based fine-tuning:

```bash
python SD_inpainting_lora_main.py \
  --data_dir ./data \
  --project_name exp_lora \
  --num_epochs 500 \
  --batch_size 1 \
  --lr 5e-7
```

**Note:** LoRA fine-tuning in the industrial PCB domain produces inconsistent results due to the domain gap between pre-trained natural images and industrial product images. See [A Note on LoRA Fine-tuning](#a-note-on-lora-fine-tuning).

### Key Insights from Experiments

Three dataset configurations (Dataset 1–3) with increasing semantic complexity were tested. Key findings:

**Quality over quantity:**  
Stable Diffusion is sensitive to semantic clarity. With precise prompts and well-annotated masks, 50 samples per class can yield stable generation. Vague semantics or many classes with very few samples causes collapse (color/texture distortion).

**Example effective class structure:**
- "A dark blue dry film residual defect" (50 samples)
- "A light blue dry film residual defect" (50 samples)
- (continue with clear color/type distinctions per class)

**Weighted sampling for imbalance:**  
If your dataset has severe class imbalance, use:

```bash
python SD_inpainting_train.py \
  --data_dir ./data \
  --use_weighted_sampler    # Upsamples minority classes each batch
```

This was critical for Dataset 3 (8 imbalanced classes): without it, minority classes produced severely distorted outputs.

---

## Inference: Generating Synthetic Defects

Once trained, generate synthetic defects:

```bash
python SD_inpainting_predict.py \
  --model_path checkpoints/exp1/best_model.pt \
  --data_dir ./data \
  --output_dir ./output
```

**Inputs (triplet per sample):**
1. **Clean base image** — defect-free PCB
2. **Binary mask** — marks region where defect should appear (white = generate here)
3. **Text prompt** — describes defect semantically ("A dark blue dry film residual defect")

**Outputs:**
- `Inpainted_results/` — final synthetic defect images
- `Combine_grid/` — side-by-side comparison grids
- `Masks/` — applied masks
- `Batch_grid/` — batch overview grids

These synthetic images are ready to mix into your classifier's training set.

**Important:** Mask shape must be semantically consistent with the prompt. A thin mask with a blob-defect prompt causes generation failure — this is a semantic mismatch, not a model issue.

---

## Results

### Augmentation Ratio vs. Recall (class: dry film residue)

| Dataset | Training samples | Recall |
|---|---|---|
| No augmentation (baseline) | 663 | 73.1% |
| 2× augmentation | 1,326 | 80.5% |
| 3× augmentation | 1,989 | 86.4% |
| **6× augmentation** | **3,978** | **91.7%** |

Scaling from 2× to 5× synthetic samples yielded up to **+19% recall** on the target class.

### Method Comparison (same 2× augmentation)

| Method | Recall |
|---|---|
| No augmentation (baseline) | 73.0% |
| GauGAN | 82.9% |
| DDPM + DIP (Method 1) | 86.9% |
| SD Inpainting (Method 2) | 86.4% |

Both diffusion-based methods significantly outperform the GAN-based GauGAN baseline.

### Training Efficiency with AMP

| Setting | GPU memory | Batch size | Training time |
|---|---|---|---|
| Standard training | ~21 GB | 1 | ~20 hours |
| AMP mixed precision | ~14 GB | 4 | ~13.5 hours |

AMP reduces training time by roughly **32%** with no perceptible difference in generation quality.

> ⚠️ **Please verify**: hardware specs above are based on the original experimental setup. Your results will vary depending on GPU.

### Cross-Domain Validation: Tray Dataset

SD Inpainting was also tested on a separate Tray dataset (5 defect classes, as few as 10 samples per class). Despite the much smaller sample count, generation quality remained high for each class — validating that the approach transfers across different industrial inspection scenarios.

---

## Dataset Design Guidelines

Based on the three-dataset experimental series (Dataset 1–3), here are the practical takeaways:

**Do:**
- Split classes by visual characteristic (color, shape) with clear, specific prompts
- Use descriptive prompts: `"A dark blue dry film residual defect"` rather than `"A blue defect"`
- Aim for at least 50 samples per class with balanced distribution across classes
- Use rough freehand masks — preserving natural edge variation helps the model learn shape diversity

**Avoid:**
- Mixing visually diverse defects into a single class (ambiguous semantics → generation collapse)
- Many classes with very few samples each (leads to color confusion and texture distortion)
- Mask shape that is semantically inconsistent with the prompt

---

## A Note on LoRA Fine-tuning

LoRA (Low-Rank Adaptation) was explored as a parameter-efficient alternative. **Results were inconsistent in the PCB defect domain.**

**Why it underperforms on industrial PCB data:**
- PCB images are highly structured industrial visuals — a significant domain gap from Stable Diffusion's natural image pre-training
- Generated defects often appeared "pasted on" (abrupt texture boundary, poor visual integration with background)
- Training loss oscillated around 0.11 (±0.02) without stable convergence
- LoRA's low-rank constraints are insufficient to bridge this domain gap

**Recommendation:** Use **full UNet fine-tuning** (not LoRA) for industrial domains. LoRA works well for natural image domains (e.g., skin lesion synthesis) where the pre-trained distribution is closer to the target.

The `SD_inpainting_lora_main.py` script is provided for reference but is not recommended for PCB defect synthesis.

---

## Project Structure

```
.
├── SD_inpainting_train.py             # UNet fine-tuning (recommended)
├── SD_inpainting_lora_main.py         # LoRA fine-tuning (alternative, not recommended)
├── SD_inpainting_predict.py           # Inference: generate synthetic defects
├── data_preprocess/
│   ├── pre_image_utils.py             # Resize images to 512×512
│   ├── pre_mask_utils.py              # Interactive mask annotation tool
│   ├── pre_prompt_utils.py            # Generate text prompts per class
│   └── final_check.py                 # Verify data completeness
├── data_loader/
│   ├── datasets.py                    # Dataset class for training
│   └── loader.py                      # DataLoader utilities
├── utils/
│   ├── load_model.py                  # Load SD Inpainting pipeline
│   ├── predict.py                     # Inference logic
│   ├── image_utils.py                 # Image I/O and transforms
│   ├── mask_utils.py                  # Mask processing
│   ├── prompt_utils.py                # Prompt generation
│   ├── plot_utils.py                  # Visualization
│   ├── save_results.py                # Save outputs
│   └── evaluate_SSIM.py               # SSIM evaluation
├── requirements.txt                   # Python dependencies
├── README.md                          # Quick start guide
└── checkpoints/                       # Trained model checkpoints
```

**Classifier training:** See [CNN Classifier Gradio Toolkit](https://github.com/Lien5757/cnn-classifier-gradio) for EfficientNet-B0 training with the synthetic defects.

---

## Citation

```bibtex
@mastersthesis{lien2025pcb,
  author  = {Lien, Yu-Hsin},
  title   = {PCB Defect Generation and Detection Techniques Based on Diffusion Models},
  school  = {Department of Industrial Engineering and Management,
             National Taipei University of Technology},
  year    = {2025}
}
```

---

## License

MIT License