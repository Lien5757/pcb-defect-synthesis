# Inference Guide

Generate synthetic defects using a trained checkpoint:

```bash
python SD_inpainting_predict.py \
  --model_path checkpoints/exp1/best_model.pt \
  --data_dir ./data \
  --output_dir ./output
```

## Parameters

| Parameter | Notes |
|---|---|
| `--model_path` | Path to trained checkpoint (e.g., `checkpoints/exp1/best_model.pt`) |
| `--data_dir` | Dataset path with `images/`, `masks/`, `texts/` subdirs |
| `--output_dir` | Where to save results |

## Outputs

Generated synthetic defect images are organized as:

- `Inpainted_results/` — Final synthetic defects
- `Combine_grid/` — Side-by-side comparisons (original + mask + result)
- `Masks/` — Applied masks
- `Batch_grid/` — Batch overview grids

## Input Format (Triplet per Sample)

The model expects three inputs per sample:

1. **Clean base image** — Defect-free PCB (512×512 PNG/JPG)
2. **Binary mask** — White (255) = generate here; Black (0) = preserve
3. **Text prompt** — Semantic description (e.g., "A dark blue dry film residual defect")

## Important

**Semantic Consistency:** Mask shape must match prompt intent.

- ✓ Thin mask + "thin line scratch" → Works
- ✗ Thin mask + "large blob defect" → Generation failure (semantic mismatch, not model bug)

If generation fails, check:
1. Does mask shape match the prompt semantically?
2. Is prompt specific and clear?
3. Was the model trained on similar prompts?

---

Next: [CNN Classifier Gradio Toolkit](https://github.com/Lien5757/cnn-classifier-gradio) for mixing synthetic defects into classifier training.  
Training: [Training Guide](training.md)
