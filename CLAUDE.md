# PCB Defect Synthesis via Stable Diffusion Inpainting

**Thesis:** PCB Defect Generation and Detection Techniques Based on Diffusion Models  
**Language:** Code in English, docs in 繁體中文 acceptable

## 🚧 Critical Constraints

### ✋ DO NOT CHANGE
- `/data`, experimental logic, model weights, Thesis.md claims

### ✅ CAN REFACTOR
- Code style, structure, docs, type hints, dependencies

### Before Any Refactor
1. Verify metrics unchanged (recall ~86.4%)
2. Confirm weighted sampler behavior identical
3. Validate checkpoint save/load works

---

## ✅ Project Status

**Phase 1: Reproducibility & Robustness** ✅
- Dependencies pinned (requirements.txt)
- Config system (TrainingConfig + JSON/CLI)
- Dynamic timestamp in output directories
- Dead code cleanup (pick_color.py, evaluate_SSIM.py)

**Phase 2: Code Quality & Maintainability** ✅
- utils/augmentation.py consolidation
- Type hints on primary files
- Google-style docstrings
- Enum for prompt_mode, config/prompts.json
- Reorganized module structure (data_loader → utils)

**Phase 3: Polish & Development Experience** ✅
- utils/validation.py (comprehensive error handling)
- Logging unification (logger everywhere)
- Config file support with validation
- Optimized plot saving (configurable interval)
- Simplified save_results with config-driven naming

**Phase 4: Documentation Refinement & Metadata** ✅
- Metadata tracking (run IDs, config.json, metadata.json)
- Refactored docs (data_prep, training, inference guides)
- Config-based training workflow documentation
- Enhanced dataset_design.md with visual examples

**Current:** Ready for GitHub release

---

## 🔗 Key References

### Audit & Analysis
- [CODEBASE_AUDIT.md](CODEBASE_AUDIT.md) — Full inventory + issues
- [Thesis.md](Thesis.md) — Methods, results, claims

### Documentation
- [docs/data_preparation.md](docs/data_preparation.md) — Data prep how-to
- [docs/training.md](docs/training.md) — Training parameters
- [docs/inference.md](docs/inference.md) — Inference how-to

### Core Code  
- [SD_inpainting_train.py](SD_inpainting_train.py) — Training entry point
- [SD_inpainting_predict.py](SD_inpainting_predict.py) — Inference entry point
- [utils/loader.py](utils/loader.py) — Data loading + weighted sampler
- [utils/datasets.py](utils/datasets.py) — PyTorch Dataset implementation

---

## 🎓 Thesis Alignment Guide

**Before finalizing ANY refactor, verify:**

✅ **Metrics Match**
- Recall on dry film: still ~91.7% at 6× augmentation?
- SSIM or other metrics unchanged?

✅ **Logic Preserved**
- Weighted sampler behavior identical?
- Loss calculation same (even if refactored)?
- Checkpoint save/load format compatible?

✅ **Data Untouched**
- No modifications to images, masks, or prompts
- Data triplet structure still intact
- final_check.py still passes

✅ **Documentation Aligned**
- Claims in Thesis.md still hold
- Method 2 (SDI) remains primary approach
- LoRA remains noted as underperforming

---

## 📋 Latest Session Log

**2026-07-03** — Phase 4 Complete  
✅ Metadata tracking (run IDs, config.json, metadata.json)  
✅ Documentation refinement (data prep, training, inference)  
✅ Repository ready for GitHub release
