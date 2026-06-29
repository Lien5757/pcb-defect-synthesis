# PCB Defect Synthesis via Stable Diffusion Inpainting — Session Guide

## 📋 Project Overview

**Thesis:** PCB Defect Generation and Detection Techniques Based on Diffusion Models  
**Repository Purpose:** Organize master's thesis experimental code for GitHub publication  
**Primary Method:** SD Inpainting (UNet fine-tuning) for synthetic PCB defect generation  
**Language:** Code in English, docs in 繁體中文 acceptable for README/comments  

### Core Files
- **Training:** `SD_inpainting_train.py` (247 LOC)
- **Inference:** `SD_inpainting_predict.py` (136 LOC)  
- **Data Prep:** 4 scripts in `data_preprocess/` (357 LOC)
- **Utils:** 9 files in `utils/` (666 LOC)

---

## 🚧 Critical Constraints

### ✋ DO NOT CHANGE
- ❌ `/data` folder (training data, masks, prompts)
- ❌ Experimental logic or loss functions (must match thesis results)
- ❌ Model checkpoints or trained weights
- ❌ Thesis.md results/claims (methodology in thesis must align with code)

### ✅ CAN REFACTOR
- Code style, structure, and organization
- Configuration management (add Config class)
- Documentation and type hints
- Utility consolidation (no functional change)
- Dependencies (pin versions in requirements.txt)

### Code = Thesis Alignment Check
Before finalizing any refactor:
1. Verify output metrics haven't changed
2. Check that SD Inpainting recall remains ~86.4%
3. Confirm weighted sampler behavior is identical
4. Validate checkpoint save/load still works

---

## 📊 Codebase Health & Progress

### Current Status: Phase 0 (Documentation + Analysis)
```
✅ Data preparation guide (docs/data_preparation.md)
✅ Training reference (docs/training.md)
✅ Inference reference (docs/inference.md)
✅ Codebase audit (CODEBASE_AUDIT.md)
```

### Code Health Scorecard
| Metric | Score | Priority |
|---|---|---|
| Functionality | 9/10 | ✅ Works |
| Maintainability | 5/10 | 🔴 Phase 1-2 |
| Reproducibility | 6/10 | 🔴 Phase 1 |
| Documentation | 8/10 | 🟡 Phase 2 |

---

## 🎯 Optimization Phases (Do One Phase Per Session)

### PHASE 1: Critical (Reproducibility & Robustness)
**Goal:** Fix breaking issues that hurt reproducibility  
**Estimated Effort:** 2-3 sessions  
**Exit Criteria:** All checkboxes done + tests pass

- [ ] **1.1** Update `requirements.txt` with pinned versions
  - [ ] Add versions for: torchvision, diffusers, opencv-python, numpy, etc.
  - [ ] Test: `pip install -r requirements.txt` works
  - [ ] Test: `python SD_inpainting_train.py` still trains

- [ ] **1.2** Create `config/train_config.py` (Config class)
  - [ ] Replace 11 scattered parameters in `SD_inpainting_train.py`
  - [ ] Support CLI: `python SD_inpainting_train.py --config config/exp1.json`
  - [ ] Backward compatible: CLI args override config file
  - [ ] Test: Old way still works

- [ ] **1.3** Fix hardcoded timestamp in `SD_inpainting_predict.py:32`
  - [ ] Change `"20221118"` → `datetime.now().strftime("%Y%m%d_%H-%M-%S")`
  - [ ] Test: output dir uses current date

- [ ] **1.4** Implement Early Stopping in `SD_inpainting_train.py`
  - [ ] Set patience threshold (default: 50 epochs)
  - [ ] Stop training when no improvement for N epochs
  - [ ] Test: training stops before num_epochs on stable loss

- [ ] **1.5** Clarify `utils/pick_color.py` purpose
  - [ ] If dead code → delete (120 LOC saved)
  - [ ] If used → document when/why + add to main pipeline
  - [ ] Same for `utils/evaluate_SSIM.py` (48 LOC)

**Validation Checklist:**
- [ ] Train on test dataset → recall within 1% of thesis claim
- [ ] Inference generates reasonable outputs
- [ ] All data is properly paired (final_check.py passes)

---

### PHASE 2: Important (Code Quality & Maintainability)
**Goal:** Improve readability and reduce technical debt  
**Estimated Effort:** 3-4 sessions  
**Prerequisite:** Complete Phase 1  
**Exit Criteria:** Type hints on all files + docstrings on public APIs

- [ ] **2.1** Create `utils/augmentation.py`
  - [ ] Consolidate: `random_flip_rotate_pil()`, `random_transform()`, crop logic
  - [ ] Remove duplicates from `loader.py` and `image_utils.py`
  - [ ] Separate data loading from augmentation in `loader.py`
  - [ ] Test: augmented data identical to before

- [ ] **2.2** Add type hints to all files
  - [ ] Use: `from typing import List, Dict, Tuple, Optional`
  - [ ] Annotate: function params, return types, class attributes
  - [ ] Tools: `mypy --strict` should pass (or with reasonable ignores)
  - [ ] Start with: train/predict main scripts

- [ ] **2.3** Add docstrings (Google style)
  - [ ] Classes: 1-2 line purpose
  - [ ] Public functions: param types, return type, 1 example
  - [ ] Private functions: minimal (internal only)
  - [ ] Tools: `pydoc` should render nicely

- [ ] **2.4** Remove unused code
  - [ ] Delete `compute_weights()` from `loader.py` (6 LOC)
  - [ ] Remove commented loss alternatives (4 LOC)
  - [ ] Remove commented Tray dataset loop from predict (16 LOC)
  - [ ] Verify tests still pass

- [ ] **2.5** Use Enum for prompt_mode
  - [ ] Replace string `'multi'`/`'single'` with `PromptMode.MULTI` / `PromptMode.SINGLE`
  - [ ] Benefits: type safety, IDE autocomplete, catch typos
  - [ ] File: `SD_inpainting_predict.py`

- [ ] **2.6** Separate data loading from augmentation
  - [ ] Create: `load_data()` and `apply_augmentation()` as separate steps
  - [ ] Update: `loader.py` to call both sequentially
  - [ ] Benefits: easier testing, reusable augmentation

**Validation Checklist:**
- [ ] `mypy` passes with max 5 ignores
- [ ] All public APIs have docstrings
- [ ] No commented code blocks remain
- [ ] Train/predict still produce identical outputs

---

### PHASE 3: Polish (Development Experience)
**Goal:** Professional setup for GitHub  
**Estimated Effort:** 2-3 sessions  
**Prerequisite:** Complete Phase 1-2  
**Exit Criteria:** Ready for `git push` to public repo

- [ ] **3.1** Add error handling & validation
  - [ ] Check data paths exist before training
  - [ ] Validate model checkpoint format
  - [ ] Handle edge cases: empty dataset, missing mask, etc.
  - [ ] Raise meaningful errors, not generic crashes

- [ ] **3.2** Unify logging (use logger everywhere)
  - [ ] Train: `logger.info()` ✓ (already done)
  - [ ] Predict: replace `print()` → `logger.info()`
  - [ ] Data prep: add logging for progress
  - [ ] Format: `[%(asctime)s] %(levelname)s: %(message)s`

- [ ] **3.3** Support config file (JSON/YAML)
  - [ ] Accept: `python SD_inpainting_train.py --config path/to/config.json`
  - [ ] Format example:
    ```json
    {
      "data_dir": "./data",
      "project_name": "exp1",
      "num_epochs": 500,
      "batch_size": 1,
      "lr": 5e-7,
      "use_weighted_sampler": true
    }
    ```
  - [ ] Test: config file + CLI override both work

- [ ] **3.4** Optimize plot saving
  - [ ] Change: save plot every epoch → every N epochs (default 10)
  - [ ] Add parameter: `--plot_interval 10`
  - [ ] Benefit: faster training, reduced disk I/O
  - [ ] Test: loss.png still captures full training curve

- [ ] **3.5** Document or remove `pick_color.py`
  - [ ] If keeping: move to `utils/` + add to main pipeline
  - [ ] If removing: delete file + update docs
  - [ ] Same for `evaluate_SSIM.py`

- [ ] **3.6** Update README with refined instructions
  - [ ] Quick start: 5 min setup
  - [ ] Config examples: different hardware/dataset sizes
  - [ ] Troubleshooting: common errors
  - [ ] Link to: docs/training.md, docs/inference.md, docs/data_preparation.md

**Validation Checklist:**
- [ ] `python SD_inpainting_train.py --help` shows all options with descriptions
- [ ] No `print()` statements in non-interactive code
- [ ] Config file validation catches bad inputs
- [ ] Plots update correctly without slowdown

---

## 📝 Session Template

**When starting a session:**

1. **Identify Phase**  
   Am I working on Phase 1, 2, or 3?

2. **Check Constraints**  
   Does my change touch `/data` or experimental logic?

3. **Read Affected Code**  
   Which files will I modify? Read them first.

4. **Write Tests First** (if possible)  
   What should stay the same? What should improve?

5. **Implement + Validate**  
   Make change → test → verify thesis alignment

6. **Document Changes**  
   Update this CLAUDE.md with progress

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
- [data_loader/loader.py](data_loader/loader.py) — Data + weighted sampler

---

## 📌 Known Issues (From Audit)

### Phase 1 Blockers
- ⚠️ Dependencies not pinned (9 packages)
- ⚠️ 11 training parameters scattered (no config system)
- ⚠️ Hardcoded timestamp "20221118" in predict
- ⚠️ Early stopping flag exists but doesn't stop training
- ⚠️ Unknown purpose of pick_color.py (120 LOC)

### Phase 2 Cleanup
- ⚠️ No type hints across codebase
- ⚠️ No docstrings on functions/classes
- ⚠️ Duplicate augmentation logic (image_utils + loader)
- ⚠️ Unused `compute_weights()` in loader
- ⚠️ Prompt mode as string (typo-prone)

### Phase 3 Polish
- ⚠️ Inconsistent logging (logger vs print)
- ⚠️ Plot saved every epoch (slow for long training)
- ⚠️ No CLI config file support
- ⚠️ Limited error handling for missing data

---

## 💾 Session Checklist

After each session, update this:

```markdown
### Session [Date]
**Phase:** [1/2/3]  
**Completed:**
- [ ] Task 1.1 / 2.x / 3.x  
- [ ] Tests passed

**Issues Found:**
- (List any new problems)

**Next Session:**
- Start with Task [?]
```

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

**Last Updated:** 2026-06-25  
**Status:** Ready for Phase 1 work  
**Next Action:** Start with Phase 1.1 (pin dependencies)
