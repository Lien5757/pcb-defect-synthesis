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

See **[Data Preparation Guide](docs/data_preparation.md)** for detailed step-by-step instructions, including:
- Image resizing (512×512)
- Interactive mask annotation
- Text prompt generation
- Data completeness verification
- Dataset design best practices

Quick summary:
```
data/
├── images/           # Defect images (organized by class)
├── masks/            # Binary masks (white = defect region)
└── texts/            # Text prompts per class
```

**Mask format:** Binary PNG. White (255) = region to synthesize defect into; Black (0) = preserve as-is. Freehand annotation is sufficient.

---

## Training the SD Inpainting Model

See **[Training Guide](docs/training.md)** for detailed parameters and setup.

Key points:
- Fine-tune the full UNet component (LoRA underperforms on industrial PCB data)
- Typical command: `python SD_inpainting_train.py --data_dir ./data --project_name exp1 --num_epochs 500`
- GPU: ~20–23 GB VRAM
- For imbalanced data, use `--use_weighted_sampler` flag
- Early stopping applied based on validation loss

See [A Note on LoRA Fine-tuning](#a-note-on-lora-fine-tuning) for why full UNet is recommended.

---

## Inference: Generating Synthetic Defects

See **[Inference Guide](docs/inference.md)** for detailed parameters and usage.

Basic command:
```bash
python SD_inpainting_predict.py \
  --model_path checkpoints/exp1/best_model.pt \
  --data_dir ./data \
  --output_dir ./output
```

**Inputs:** Clean image + binary mask + text prompt  
**Outputs:** Synthetic defect images in `Inpainted_results/`, comparison grids, and masks

**Important:** Mask shape must match prompt semantically. Thin mask + "large blob" prompt = failure (semantic mismatch, not model bug).

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

For dataset design best practices and guidelines, see [Data Preparation Guide](docs/data_preparation.md).

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