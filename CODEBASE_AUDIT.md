# Codebase Audit Report

**Generated:** 2026-06-25  
**Total Files:** 17 Python + 2 config  
**Total Lines of Code:** 1,582

---

## 1. Repository Structure & File Purposes

### 📊 Core Training & Inference (2 files, 383 LOC)

| File | Lines | Purpose | Status |
|---|---|---|---|
| **SD_inpainting_train.py** | 247 | UNet fine-tuning on PCB defect data | ✅ Core |
| **SD_inpainting_predict.py** | 136 | Batch inference to generate synthetic defects | ✅ Core |

---

### 🔧 Data Processing (4 files, 357 LOC)

| File | Lines | Purpose | Status |
|---|---|---|---|
| **data_preprocess/pre_image_utils.py** | 81 | Image resize/rename utilities | ✅ Prod |
| **data_preprocess/pre_mask_utils.py** | 187 | Interactive mask annotation tool (draw_np_v2) | ✅ Prod |
| **data_preprocess/pre_prompt_utils.py** | 51 | Generate text prompts from class-to-prompt mapping | ✅ Prod |
| **data_preprocess/final_check.py** | 38 | Verify data triplet completeness | ✅ Prod |

---

### 📦 Data Loading (2 files, 176 LOC)

| File | Lines | Purpose | Status |
|---|---|---|---|
| **data_loader/datasets.py** | 89 | `InpaintingDataset` class for training data | ✅ Core |
| **data_loader/loader.py** | 87 | DataLoader + weighted sampler logic | ✅ Core |

---

### 🛠 Utilities (9 files, 666 LOC)

| File | Lines | Purpose | Status | Issues |
|---|---|---|---|---|
| **utils/load_model.py** | 21 | Load SD Inpainting pipeline + scheduler | ✅ Core | None |
| **utils/predict.py** | 32 | Inference batch prediction logic | ✅ Core | None |
| **utils/save_results.py** | 127 | Save inpainted outputs (grid + individual) | ✅ Core | Complex save logic |
| **utils/plot_utils.py** | 72 | Training visualization (loss, lr plots) | ✅ Prod | Saves every epoch (slow) |
| **utils/image_utils.py** | 67 | Image I/O, augmentation (flip, rotate) | ✅ Prod | None |
| **utils/mask_utils.py** | 113 | Mask transforms (random crops, affine) | ✅ Prod | Similar to loader.py |
| **utils/prompt_utils.py** | 66 | Hardcoded prompt configs per experiment | ⚠️ Prod | Hardcoded exp names |
| **utils/evaluate_SSIM.py** | 48 | SSIM evaluation metric | ⚠️ Unused | Unknown usage |
| **utils/pick_color.py** | 120 | Color picker utility (unclear purpose) | ❓ Unknown | Dead code? |

---

### 📄 Configuration & Documentation (6 files)

| File | Purpose | Status |
|---|---|---|
| **requirements.txt** | Python dependencies | ✅ Complete |
| **README.md** | Main documentation | ✅ Updated |
| **Thesis.md** | High-level methodology & results | ✅ Updated |
| **docs/data_preparation.md** | Step-by-step data prep guide | ✅ Added |
| **docs/training.md** | Training parameters reference | ✅ Added |
| **docs/inference.md** | Inference parameters reference | ✅ Added |

---

## 2. Code Classification

### 🎯 Core Experimental Code (Essential for thesis)

**Training Pipeline:**
- `SD_inpainting_train.py` — UNet fine-tuning (main result)
- `data_loader/datasets.py` + `data_loader/loader.py` — Training data loading
- `utils/load_model.py` — Pipeline initialization

**Inference Pipeline:**
- `SD_inpainting_predict.py` — Synthetic defect generation
- `utils/predict.py` — Batch prediction
- `utils/save_results.py` — Output organization

**Data Preparation:**
- `data_preprocess/pre_image_utils.py` — Image standardization
- `data_preprocess/pre_mask_utils.py` — Mask annotation (manual)
- `data_preprocess/pre_prompt_utils.py` — Prompt generation
- `data_preprocess/final_check.py` — Validation

**Result: 11 files, 1,092 LOC**

---

### 🔧 Utility/Support Code

**Visualization & Monitoring:**
- `utils/plot_utils.py` — Training metrics visualization
- `utils/image_utils.py` — Image augmentation helpers

**Supporting Logic:**
- `utils/mask_utils.py` — Mask transformation
- `utils/prompt_utils.py` — Prompt mapping

**Result: 4 files, 318 LOC**

---

### ❓ Unclear/Unused Code

| File | Purpose | Recommendation |
|---|---|---|
| **utils/pick_color.py** | Color picker (120 LOC) | Clarify or remove |
| **utils/evaluate_SSIM.py** | SSIM calculation (48 LOC) | Not used in main pipeline |
| **SD_inpainting_lora_main.py** | LoRA fine-tuning | Removed from docs (underperforms) |

---

## 3. Dead Code & Redundancies

### 🔴 Unused Functions

| Location | Function | LOC | Status |
|---|---|---|---|
| **data_loader/loader.py** | `compute_weights()` | 6 | Only `compute_soft_weights()` is used |
| **SD_inpainting_train.py** | Commented loss calculation | 4 | Lines 192-195: alternative loss logic |
| **SD_inpainting_predict.py** | Commented Tray dataset loop | 16 | Lines 119-135: unused experiment |

### 🟡 Duplicated Logic

| Issue | File 1 | File 2 | Recommendation |
|---|---|---|---|
| **Image transforms** | `image_utils.py:random_flip_rotate_pil()` | `loader.py:ImageOnlyTransform` | Consolidate to augmentation.py |
| **Mask processing** | `mask_utils.py` | `loader.py:ImageOnlyTransform` | Separate augmentation from loading |
| **Prompt selection** | `predict.py:set_prompts()` | `prompt_utils.py` | Duplicate prompt definitions |

### 🟠 Unclear Code Purpose

| File | Purpose | Clarification Needed |
|---|---|---|
| **pick_color.py** | Color selection utility | When/why is this used? (120 LOC) |
| **evaluate_SSIM.py** | Similarity metric | Used for validation? Results? |

---

## 4. Dependency Audit

### ✅ Pinned Versions

```
torch==2.4.1 ✓ (critical for CUDA 11.8)
transformers==4.46.3 ✓
```

### ⚠️ Unpinned Versions (May break)

| Package | Status | Concern |
|---|---|---|
| torchvision | No version | Should match torch==2.4.1 |
| diffusers | No version | Breaking changes possible |
| peft | No version | LoRA compatibility |
| opencv-python | No version | API changes possible |
| matplotlib | No version | Plot compatibility |
| scikit-image | No version | Transform API changes |
| tqdm | No version | Minor (usually stable) |
| pillow | No version | Image format handling |
| numpy | No version | Critical dependency |

### 📋 Requirement Completeness

**Missing from requirements.txt:**
- `argparse` — Built-in, not needed
- `logging` — Built-in, not needed
- `json` — Built-in, not needed

**Recommendation:** Pin all versions to ensure reproducibility:
```
torch==2.4.1
torchvision==0.19.1  # Matches torch 2.4.1
diffusers==0.25.0
transformers==4.46.3
peft==0.10.0
opencv-python==4.8.1
matplotlib==3.8.0
scikit-image==0.22.0
tqdm==4.66.0
pillow==10.1.0
numpy==1.24.3
```

---

## 5. Architecture & Design Issues

### 🔴 High Priority Issues

| Issue | Severity | Impact | Files |
|---|---|---|---|
| **No Config Management** | HIGH | 11 CLI parameters scattered | SD_inpainting_train.py |
| **Hardcoded Paths** | HIGH | Non-reproducible | predict.py, prompt_utils.py |
| **No Type Hints** | HIGH | Unclear interfaces | All .py files |
| **Missing Early Stopping** | HIGH | Wasted training time | SD_inpainting_train.py:224 |
| **Hardcoded Timestamp** | HIGH | Fixed to 2022-11-18 | SD_inpainting_predict.py:32 |

### 🟡 Medium Priority Issues

| Issue | Severity | Impact | Files |
|---|---|---|---|
| **Coupling: Data Loading + Augmentation** | MEDIUM | Hard to test | loader.py |
| **Prompt Mode as String** | MEDIUM | Typo-prone | SD_inpainting_predict.py:87 |
| **No Docstrings** | MEDIUM | Unclear intent | All .py files |
| **Inconsistent Logging** | MEDIUM | Debug difficult | train: logger, predict: print |
| **Loss Function Hardcoded** | MEDIUM | Can't experiment | SD_inpainting_train.py:198 |

### 🟢 Low Priority Issues

| Issue | Severity | Impact | Files |
|---|---|---|---|
| **Plot Saved Every Epoch** | LOW | IO overhead | plot_utils.py |
| **Device String Hardcoded** | LOW | No multi-GPU support | SD_inpainting_train.py:40 |
| **Magic Numbers** | LOW | Maintainability | loader.py, SD_inpainting_train.py |

---

## 6. Code Quality Metrics

### Size Distribution

```
├── Core Training & Inference       383 LOC (24%)
├── Data Processing                 357 LOC (23%)
├── Utilities                       666 LOC (42%)
└── Data Loading                    176 LOC (11%)
```

### Complexity Hotspots

| File | Complexity | Main Issue |
|---|---|---|
| **utils/save_results.py** | 127 LOC | Complex grid/individual save logic |
| **utils/mask_utils.py** | 113 LOC | Multiple transform options |
| **utils/pick_color.py** | 120 LOC | Purpose unclear |
| **data_preprocess/pre_mask_utils.py** | 187 LOC | Interactive GUI (acceptable) |

### Function Distribution

- **Total Functions:** 39
- **Avg Lines per Function:** ~41 LOC
- **Largest Function:** `draw_np_v2()` (157 LOC, acceptable for interactive tool)

---

## 7. Recommendations (Priority Order)

### Phase 1: Critical (Do Now)

- [ ] **1.1** Pin all dependencies in requirements.txt
- [ ] **1.2** Create `Config` class to replace CLI parameters
- [ ] **1.3** Fix hardcoded timestamp in predict.py (use datetime.now())
- [ ] **1.4** Implement real Early Stopping logic
- [ ] **1.5** Clarify purpose of `pick_color.py` (remove if dead code)

### Phase 2: Important (Next)

- [ ] **2.1** Create `utils/augmentation.py` (consolidate transforms)
- [ ] **2.2** Add type hints to all files
- [ ] **2.3** Add docstrings to all classes/functions
- [ ] **2.4** Remove unused `compute_weights()` from loader.py
- [ ] **2.5** Use Enum for prompt_mode (not string)
- [ ] **2.6** Separate data loading from augmentation

### Phase 3: Polish (Later)

- [ ] **3.1** Add error handling & validation
- [ ] **3.2** Unify logging (use logger everywhere)
- [ ] **3.3** Support CLI config file (JSON/YAML)
- [ ] **3.4** Optimize plot saving (every N epochs)
- [ ] **3.5** Document pick_color.py purpose or remove

---

## 8. Summary

### ✅ Strengths
- Clear separation of concerns (train/predict/preprocess)
- Well-documented high-level overview (README, Thesis.md)
- Comprehensive data preprocessing pipeline
- Good checkpoint/resume logic

### ⚠️ Weaknesses
- No configuration management system
- Scattered hardcoded values (paths, timestamps, params)
- Dead/unused code not cleaned up
- Inconsistent code style (camelCase vs snake_case)
- Missing type hints and docstrings

### 📈 Code Health: 6.5/10
- Functionality: 9/10 (works as intended)
- Maintainability: 5/10 (needs cleanup)
- Reproducibility: 6/10 (hardcoded paths/configs)
- Documentation: 8/10 (good high-level, weak code-level)

**Next Action:** Start with Phase 1 (Critical) for immediate improvement.
