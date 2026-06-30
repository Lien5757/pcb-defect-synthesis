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
├── data_preprocess/          # Data preparation scripts
│   ├── pre_image_utils.py
│   ├── pre_mask_utils.py
│   ├── pre_prompt_utils.py
│   └── final_check.py
├── config/                   # Configuration files
│   ├── train_config.py
│   ├── prompts.json
│   └── enums.py
├── utils/                    # Utility modules
│   ├── loader.py             # Data loading
│   ├── datasets.py           # PyTorch Dataset
│   ├── augmentation.py       # Image augmentation
│   ├── prompt_utils.py       # Prompt handling
│   └── ...
├── docs/                     # Documentation
│   ├── data_preparation.md
│   ├── data_annotation.md
│   ├── dataset_design.md
│   ├── training.md
│   ├── inference.md
│   └── config.md
├── SD_inpainting_train.py    # Training entry point
├── SD_inpainting_predict.py  # Inference entry point
├── requirements.txt
└── README.md
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

**Minimum:** Python 3.10, CUDA 11.8, **24 GB VRAM** (training)

See [requirements.txt](requirements.txt) for full dependency list.

### Hardware

| Task | VRAM | Device |
|------|------|--------|
| Train (BS=1) | 20.6 GB | RTX 4090 |
| Train (BS=4) | 22.6 GB | RTX 4090 |
| Inference | ~11.6 GB | RTX 4090 |

> Inference may work on 12 GB GPUs; training requires 24 GB.

---

## Quick Start

### 1. Prepare Data

```bash
python data_preprocess/pre_image_utils.py      # Resize to 512×512
python data_preprocess/pre_mask_utils.py       # Draw masks (interactive)
python data_preprocess/pre_prompt_utils.py     # Generate text prompts
python data_preprocess/final_check.py          # Verify completeness
```

📖 **[Full Data Preparation Guide](docs/data_preparation.md)**

### 2. Train

```bash
python SD_inpainting_train.py --data_dir ./data --project_name exp1
```

📖 **[Training Guide](docs/training.md)** | 📋 **[Config Reference](docs/config.md)**

### 3. Inference

```bash
python SD_inpainting_predict.py \
  --model_path checkpoints/exp1/best_model.pt \
  --data_dir ./data
```

📖 **[Inference Guide](docs/inference.md)**

---

## Documentation

| Document | Purpose |
|----------|---------|
| [Data Preparation](docs/data_preparation.md) | Organize, annotate, prepare data |
| [Data Annotation Details](docs/data_annotation.md) | Step-by-step annotation walkthrough |
| [Dataset Design](docs/dataset_design.md) | Best practices + troubleshooting |
| [Training Guide](docs/training.md) | Training parameters and techniques |
| [Inference Guide](docs/inference.md) | Generate synthetic defects |
| [Config Reference](docs/config.md) | Training config file format |

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
