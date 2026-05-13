# SC-Captioner: Mid-Semester Evaluation Presentation
## Technical Progress & Implementation Details

---

## SLIDE 1: Problem Statement & Research Motivation

### Core Problem

**Challenge:** Image-to-text captioning models often generate incomplete or hallucinated captions.

Example:
- ❌ Initial Caption: "a dog"
- ✓ Better Caption: "a brown dog sitting on green grass in a sunny park"

**Why This Matters:**
- Accessibility: Better descriptions for visually impaired users
- Information retrieval: More accurate image-based search
- Model robustness: Understanding when models are confident vs. speculative

### Proposed Solution: Self-Correction via RL

**Key Innovation:** Enable models to **self-correct their own captions** using reinforcement learning

**Three-Stage Approach:**

```
Stage 1: Generate Initial Caption
    ↓
Stage 2: Model Attempts Self-Correction
    ↓
Stage 3: Score Corrections Using Reward Function
         (Scene-graph decomposition + element matching)
    ↓
Train Model to Maximize Correct Refinements
```

**Why This Matters:**
- Humans can self-correct; why not models?
- Direct feedback on what's "good" vs "bad" correction
- Online learning: improve from generated data

### Base Code Source

**Original Paper:** "SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning"
- **Conference:** ICCV 2025
- **Authors:** Zhang, Lin et al.
- **arXiv ID:** 2508.06125
- **Repository:** github.com/zl2048/SC-Captioner

**Foundation Framework:** LLaMA-Factory
- **Purpose:** Unified training framework for vision-language models
- **Features:** Multi-stage training, LoRA support, configuration-based workflows
- **Why Chosen:** Modular design enables SFT → Merge → SC pipeline

**Vision-Language Model:** Qwen2-VL-2B (Alibaba)
- **Lightweight:** 2 billion parameters
- **Mac-Compatible:** Can train on Apple Silicon
- **Effective:** SOTA performance for its size

---

## SLIDE 2: Virtual Environment & Dependency Compatibility

### Hardware Target & Constraints

**Development Hardware:**
```
Device:  Apple Mac M-series (16GB unified memory)
GPU:     No CUDA (uses Metal Performance Shaders instead)
Python:  3.10
Disk:    ~50GB available
```

### Virtual Environment Setup

**Step 1: Create Conda Environment**
```bash
conda create -n p3.10env python=3.10
conda activate p3.10env
```

**Step 2: Install PyTorch with MPS Backend**
```bash
conda install pytorch::pytorch torchvision -c pytorch
python -c "import torch; print(torch.backends.mps.is_available())"
# Returns: True ✓
```

**Step 3: Install Transformers & Core Dependencies**
```bash
pip install transformers==4.45.0     # Exact version (critical!)
pip install datasets>=2.16.0,<=2.21.0
pip install accelerate>=0.30.1,<=0.34.2
pip install peft>=0.11.1,<=0.12.0
pip install trl>=0.8.6,<=0.9.6 --no-deps
pip install -e .  # Install SC-Captioner in dev mode
```

### Dependency Compatibility Matrix

| Component | Version | Platform | Status |
|-----------|---------|----------|--------|
| Python | 3.10.x | Mac | ✓ Required |
| PyTorch | 2.x | MPS backend | ✓ Validated |
| Transformers | 4.45.0 | CPU/MPS | ✓ **Exact** |
| Accelerate | 0.34.2 | MPS support | ✓ Critical |
| PEFT | 0.12.0 | LoRA | ✓ Tested |
| TRL | 0.12.0 | DPO trainer | ✓ Custom |
| Datasets | 2.21.0 | Data loading | ✓ Tested |

### Key Compatibility Checks Performed

#### 1. **CUDA vs. MPS** (Apple Silicon requires MPS)
```python
import torch
print(f"MPS Available: {torch.backends.mps.is_available()}")
# Output: True ✓ (no CUDA on Mac)
```

#### 2. **Precision Support** (MPS supports BF16, NOT FP16)
| Precision | NVIDIA GPU | Apple MPS | Our Decision |
|-----------|-----------|-----------|-------------|
| FP32 | ✓ | ✓ | Fallback |
| **BF16** | ✓ | **✓** | **Selected** |
| FP16 | ✓ | ❌ | Avoided |

**Why BF16?** Preserves gradient range (like FP32) while using half memory (like FP16)

#### 3. **Version Pinning**
```
Why Exact Versions?
- TRL 0.12.0 API expects Transformers 4.45.0
- Accelerate 0.34.2 first MPS-optimized release
- Mixing versions → subtle bugs at runtime
```

---

## SLIDE 3: Code Architecture & Organization

### Directory Structure

```
SC-Captioner/
├── config/                         # Training configs (YAML)
│   ├── qwen2vl_train_lora_sft.yaml
│   ├── qwen2vl_train_lora_sc_2b.yaml  ← Mac-optimized SC
│   └── qwen2vl_test_lora_sc_docci500.yaml
│
├── src/llamafactory/train/sc/      # ⭐ SC-specific code
│   ├── trainer.py                  # CustomSCTrainer (reward integration)
│   ├── workflow.py                 # SC pipeline orchestration
│   ├── reward_utils.py             # Scene-graph reward calculation
│   └── capture.py                  # Metric evaluation
│
├── data/                            # Datasets
│   ├── train_coco6k_2_mini.json    # Training captions (mini for testing)
│   ├── images/coco/train2017/      # COCO images
│   └── images/docci/               # DOCCI test images
│
├── saves/qwen2_vl-2b/              # Checkpoints
│   ├── lora/sft_coco6k_small/      # After SFT
│   ├── merged/sft_coco6k_small/    # After merge
│   └── lora/sc_coco6k_small/       # After SC (final)
│
└── [documentation files]
```

### Data Flow & Training Pipeline

```
Configuration (YAML)
        ↓
    CLI Entry
    ├─ llamafactory-cli train config.yaml
        ↓
    Stage Router
    ├─ stage: sc → SC workflow selected
        ↓
    Load Components
    ├─ Tokenizer & templates
    ├─ Dataset (initial + corrected captions)
    ├─ Base model (merged from SFT)
    ├─ Reference model (for reward KL penalty)
        ↓
    Data Processing
    ├─ Tokenize image + captions (pairwise.py)
    ├─ Collate batches (collator.py)
    ├─ Preserve raw text for reward
        ↓
    Training Loop (CustomSCTrainer)
    ├─ Generate captions
    ├─ Calculate reward (scene-graph)
    ├─ Compute DPO loss
    ├─ Backprop & update
        ↓
    Save Checkpoints
```

### Key Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Data Loader** | Load images + caption pairs | Huggingface `datasets` |
| **Preprocessor** | Tokenize captions, preserve text | `pairwise.py` |
| **Model** | Vision-language backbone | Qwen2-VL-2B + LoRA |
| **Collator** | Batch assembly with multimodal data | Custom `multimodal_sc_collator` |
| **Trainer** | Training loop with rewards | `CustomSCTrainer` (TRL-based) |
| **Reward Function** | Scene-graph matching | `reward_utils.py` |
| **Metrics** | Evaluation scoring | `capture.py` (BLEU, ROUGE, SPICE) |
| **Output** | Saved checkpoints | LoRA weights + config |

---

## SLIDE 4: Reward Function & Self-Correction Logic

### Scene-Graph Based Reward Design

**Key Insight:** Captions are compositions of objects, attributes, and relations

Example:
```
Caption: "a brown dog sitting on green grass"
├─ Objects: {dog, grass}
├─ Attributes: {brown, green, sitting}
└─ Relations: {on}
```

### Reward Calculation Algorithm

```
INPUT:
  initial_caption: "a dog on grass"
  corrected_caption: "a brown dog on green grass"
  reference_caption: "a brown dog sitting on green grass"

PARSE into scene graphs:
  initial_sg = {dog, grass} / {brown, green, sitting} / {on}
  corrected_sg = {dog, grass} / {brown, green} / {on}
  reference_sg = {dog, grass} / {brown, green, sitting} / {on}

COMPUTE DELTA (what changed):
  added = corrected_sg - initial_sg = {brown}
  removed = initial_sg - corrected_sg = {}

SCORE:
  For each added element:
    - If in reference: +1.0 (good correction)
    - If not in reference: -1.0 (hallucination)
  
  For each removed element:
    - If not in reference: +0.5 (good removal)
    - If in reference: -0.5 (bad removal)

  reward = sum_of_scores / max(|reference|, 1)

OUTPUT:
  reward ≈ 0.5 (partial credit: added "brown" but missed "sitting")
```

### Online DPO Training Integration

```
Traditional DPO:
├─ Requires: {prompt, good_response, bad_response}
├─ Source: Static dataset
└─ Problem: Need manual annotation for all pairs

Our Online DPO:
├─ Generate: initial caption (baseline)
├─ Generate: corrected caption (from model after training)
├─ Score: both using reward function
├─ Treat: reward_corrected > reward_initial as preference
└─ Benefit: Automatic pair generation from rewards
```

---

## SLIDE 5: Problems Faced & Solutions

### Critical Issues

#### Problem 1: Out-of-Memory (OOM) During Generation
**Severity:** 🔴 Critical | **Status:** ✅ Fixed

**Issue:**
- Config had `max_new_tokens: 512`
- Each training step generates 2 captions (initial + corrected)
- Exceeds 16GB Mac memory

**Solution:** Reduce to `max_new_tokens: 128`
```yaml
# config/qwen2vl_train_lora_sc_2b.yaml
generation_config:
  max_new_tokens: 128  # ⬇️ From 512
```
**Impact:** ✓ Training runs to completion (41 minutes)

#### Problem 2: Training Crashes at Completion
**Severity:** 🔴 Critical | **Status:** ✅ Fixed

**Issue:**
```
TypeError: save() got an unexpected keyword argument 'license'
```
- TRL calls `model_card.save(license="apache-2.0")`
- Huggingface ModelCard doesn't have `license` parameter

**Solution:** Remove invalid parameter
```python
# src/llamafactory/train/trainer_utils.py
# BEFORE: model_card.save(card_data=model_card_data, license="apache-2.0")
# AFTER:  model_card.save(card_data=model_card_data)
```
**Impact:** ✓ Checkpoints save successfully

#### Problem 3: Multiprocessing Hangs with MPS
**Severity:** 🟠 High | **Status:** ✅ Fixed

**Issue:** DataLoaders with `num_workers > 2` freeze on MPS backend

**Solution:** Set `preprocessing_num_workers: 2` in config
```yaml
preprocessing_num_workers: 2  # Safe for MPS
```
**Impact:** ✓ Data loading stable

#### Problem 4: Unsupported Precision Format
**Severity:** 🟠 High | **Status:** ✅ Fixed

**Issue:** FP16 not supported on MPS backend

**Solution:** Use BF16 instead
```yaml
fp16: false
bf16: true  # ✓ MPS-compatible, preserves gradient stability
```
**Impact:** ✓ 50% memory reduction vs FP32

### Summary Table

| Issue | Root Cause | Fix | Status |
|-------|-----------|-----|--------|
| Generation OOM | Large `max_new_tokens` | Reduce to 128 | ✅ |
| Training crash | Invalid TRL API | Remove `license=` param | ✅ |
| Multiprocessing hang | MPS constraints | Set `num_workers: 2` | ✅ |
| FP16 error | MPS unsupported | Use `bf16: true` | ✅ |
| Memory overflow | Long sequences | Reduce `cutoff_len` 4096→1024 | ✅ |

---

## SLIDE 6: Training Results & Current Status

### Training Execution Summary

**Configuration:** `config/qwen2vl_train_lora_sc_2b.yaml` (Mac-optimized)

| Metric | Value |
|--------|-------|
| Model | Qwen2-VL-2B |
| Training Method | LoRA + Online DPO |
| Duration | 41 minutes |
| Training Steps | 9 completed ✓ |
| Hardware | Mac M1/M2/M3 (16GB RAM) |
| Batch Size | 1 (per device) |
| Gradient Accumulation | 2 steps |
| Effective Batch | 2 samples/update |
| Precision | BF16 |
| Memory Status | ✓ Stable (no OOM) |

### Training Checkpoints

```
saves/qwen2_vl-2b/lora/sc_coco6k_small/
├── checkpoint-9/              # Final checkpoint
│   ├── adapter_model.bin      # LoRA weights
│   ├── adapter_config.json
│   └── training_args.bin
├── adapter_model.bin          # Best checkpoint
├── trainer_state.json         # Training logs
└── README.md                  # Model card
```

**Status:** ✓ Checkpoints saved successfully

### Mac Stability Indicators

✓ Memory usage stable throughout training  
✓ No out-of-memory crashes  
✓ No multiprocessing deadlocks  
✓ No precision format errors  
✓ Checkpoint integrity verified  

---

## SLIDE 7: Work Completed vs. Remaining

### Semester Progress

#### ✅ Completed (First Half)

1. **Environment Setup**
   - Python 3.10 + conda environment
   - PyTorch with MPS backend validation
   - Full dependency stack compatibility checks
   - Documentation of all version constraints

2. **Code Adaptation**
   - Fixed TRL trainer for Mac (model card save)
   - Optimized generation for memory constraints
   - Reduced sequence length for 16GB hardware
   - Validated precision support (BF16)

3. **Training Execution**
   - Completed SFT (Supervised Fine-Tuning) on COCO6K
   - Merged LoRA weights into base model
   - Initiated SC training on mini dataset
   - 9 steps completed successfully
   - 41-minute stable training run

4. **Documentation**
   - Technical setup documentation
   - Code flow explanation (Understand.md)
   - Implementation notes (Summary.md)
   - Presentation materials

#### ⏳ Remaining (Second Half)

1. **Scale Training** (1-2 weeks)
   - Expand from mini to full RefinedCaps dataset
   - Estimate: ~4-6 hours training on full dataset
   - Monitor memory behavior at scale

2. **Evaluation** (1-2 weeks)
   - Generate predictions on DOCCI500 (500 test images)
   - Compute metrics: BLEU, ROUGE, METEOR, CIDEr, SPICE, CAPTURE
   - Create comparison vs. baseline (SFT only)
   - Scene-graph accuracy analysis

3. **Validation & Analysis** (1-2 weeks)
   - Case studies: show improved captions
   - Ablation studies: reward function effectiveness
   - Error analysis: failure modes
   - Performance tables

4. **Finalization** (Final week)
   - Write technical report
   - Prepare final presentation
   - Code cleanup & documentation
   - Submit for evaluation

### Timeline

```
Feb-Mar 2026:  Setup & Bug Fixes ============== ✅ Complete
Mar-Apr 2026:  Training & Optimization ======== 🟢 In Progress
Apr-May 2026:  Evaluation & Analysis ========== ⏳ Pending
May 2026:      Final Report & Presentation ==== ⏳ Pending
```

---

## SLIDE 8: Key Learnings & Next Steps

### Technical Insights Gained

1. **Hardware-Specific Tuning**
   - Apple Silicon ≠ CUDA (completely different backend)
   - Precision, workers, sequence length all must be tuned
   - No one-size-fits-all config exists

2. **Version Dependencies Matter**
   - TRL 0.12.0 + Transformers 4.45.0 = specific combo
   - Mixing versions causes subtle runtime bugs
   - Must validate entire stack together

3. **Modular Pipeline Design**
   - SFT → Merge → SC stages enable incremental validation
   - Each stage produces usable checkpoint
   - Easier to debug specific stages

4. **Memory Bottleneck**
   - Training forward pass << generation memory
   - KV-cache during generation is critical constraint
   - Reduction in max_new_tokens had highest impact

### Next Immediate Steps

**Week 1-2: Scale Training**
```bash
# Switch to full dataset
sed 's/train_coco6k_2_mini/train_coco6k_2/g' config/qwen2vl_train_lora_sc_2b.yaml
llamafactory-cli train config/qwen2vl_train_lora_sc_2b.yaml
```

**Week 3-4: Evaluation**
```bash
# Generate predictions
llamafactory-cli train config/qwen2vl_test_lora_sc_docci500.yaml

# Compute metrics
./run_metrics_docci500.sh saves/eval_qwen2vl/sc/docci500
```

**Week 5-6: Analysis**
- Compare metrics: SC-trained vs. SFT-only baseline
- Analyze which caption types improve most
- Document failure cases

### Success Criteria

✓ Training completes on full dataset (no OOM)  
✓ Metrics show improvement over baseline  
✓ CAPTURE metric validates scene-graph reward  
✓ Code is reproducible on Mac hardware  
✓ Documentation is comprehensive  

---

## Additional Resources

### Key Files in Repository

| File | Purpose |
|------|---------|
| [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) | Detailed technical reference |
| [Summary.md](Summary.md) | Implementation notes |
| [Understand.md](Understand.md) | Code flow explanation |
| [config/qwen2vl_train_lora_sc_2b.yaml](config/qwen2vl_train_lora_sc_2b.yaml) | Mac-optimized training config |

### Quick Commands

```bash
# Activate environment
conda activate p3.10env
cd /Users/ektajangid/source_code/mtech_project/SC-Captioner

# Run training
llamafactory-cli train config/qwen2vl_train_lora_sc_2b.yaml

# Generate predictions
llamafactory-cli train config/qwen2vl_test_lora_sc_docci500.yaml

# Evaluate metrics
./run_metrics_docci500.sh saves/eval_qwen2vl/sc/docci500
```

### Contact & Questions

**Current Status:** All critical issues resolved; training stable  
**Next Phase:** Scale to full dataset & evaluation  
**Timeline:** On track for end-of-semester submission  

---

## References

1. **Original Paper:** Zhang et al., "SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning", ICCV 2025
2. **Base Framework:** LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory)
3. **Vision Model:** Qwen2-VL (https://qwenlm.github.io/blog/qwen2-vl/)
4. **Training Library:** TRL (https://github.com/huggingface/trl)
5. **Hardware:** Apple Silicon MPS (https://pytorch.org/blog/introducing-accelerated-gpu-training-on-mac/)
