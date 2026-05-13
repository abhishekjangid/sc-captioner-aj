# SC-Captioner: Mid-Term Evaluation Presentation
## Capstone Project - Condensed Slides

---

## SLIDE 1: Project Overview & Problem Statement

### SC-Captioner: Improving Image Captioning with Self-Correction

**Project Goal:**  
Implement a reinforcement learning framework that enables image captioning models to **self-correct their own captions** for improved accuracy and detail.

**Base Paper:** "SC-Captioner" - ICCV 2025  
**Visual-Language Model:** Qwen2-VL-2B (Alibaba)  
**Hardware:** Mac M-series (Apple Silicon, 16GB RAM)

### The Problem We're Solving

**Challenge:** Vision-language models often miss fine-grained details or hallucinate objects in image captions.

**Our Approach:**
1. Generate initial captions from the model
2. Allow the model to **self-correct** its own captions  
3. Use **reward functions** (scene-graph based) to incentivize accurate corrections
4. Train via **Online DPO (Reinforcement Learning)**

**Why It Matters:** Better captions = better accessibility, translation, and visual understanding

---

## SLIDE 2: Work Completed (Mid-Term Progress)

### Achievements to Date

#### ✅ Environment Setup
- Configured Python 3.10 virtual environment on Mac M-series
- Resolved Apple Silicon compatibility issues (MPS backend instead of CUDA)
- Validated full dependency stack: PyTorch, Transformers 4.45.0, TRL, PEFT

#### ✅ Code Adaptation & Bug Fixes
- **Fixed Generation OOM:** Reduced `max_new_tokens` from 512 → 128
- **Fixed Training Crash:** Removed invalid `license=` parameter from TRL's model card save
- **Tuned for Mac Memory:** Set `preprocessing_num_workers: 2`, enabled `bf16` precision
- **Result:** Stable training without crashes

#### ✅ Training Execution (SFT → Merge → SC)
1. **SFT Stage:** Supervised fine-tuning on COCO6K mini dataset ✓
2. **Merge Stage:** Merged LoRA weights into base model ✓
3. **SC Stage:** Self-Correction training initiated
   - Duration: 41 minutes
   - Steps Completed: 9/9
   - Checkpoint Saved: `saves/qwen2_vl-2b/lora/sc_coco6k_small/`
   - Memory: Stable, no OOM

#### ✅ Documentation
- Comprehensive setup guide (Summary.md)
- Code flow analysis (Understand.md)
- Version compatibility matrix

---

## SLIDE 3: Technical Implementation

### Architecture: 3-Stage Training Pipeline

```
Stage 1: SFT (Supervised Fine-Tuning)
├─ Input: Qwen2-VL-2B + image-caption pairs
├─ Output: SFT checkpoint with LoRA weights
└─ Status: ✓ Complete

Stage 2: Merge
├─ Input: Base model + SFT LoRA weights
├─ Output: Merged checkpoint ready for SC
└─ Status: ✓ Complete

Stage 3: SC (Self-Correction via Online DPO)
├─ Input: Merged model + preference pairs
├─ Data: Initial captions → Self-corrected captions
├─ Reward: Scene-graph based scoring (objects, attributes, relations)
├─ Trainer: Online DPO with reward integration
└─ Status: ✓ In Progress (9 steps completed)
```

### Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| **Data Pipeline** | Tokenize image + caption pairs, preserve chosen/rejected text | ✓ Working |
| **Reward Function** | Scene-graph decomposition + element-level matching | ✓ Implemented |
| **SC Trainer** | Online DPO loss with reward calculation | ✓ Operational |
| **Config System** | Mac-optimized hyperparameters (batch_size=1, bf16, num_workers=2) | ✓ Validated |

### Configuration for Mac

```yaml
# Memory-efficient settings for 16GB Mac
fp16: false           # Not supported on MPS
bf16: true           # Use BF16 precision
batch_size: 1        # Minimal memory footprint
gradient_accumulation_steps: 2  # Effective batch = 2
preprocessing_num_workers: 2    # Avoid MPS overhead
max_new_tokens: 128  # Reduced for generation
cutoff_len: 1024     # Sequence length
```

---

## SLIDE 4: Results & Current Status

### Training Metrics

| Metric | Value |
|--------|-------|
| **Model** | Qwen2-VL-2B |
| **Adaptation** | LoRA (parameter-efficient) |
| **Training Duration** | 41 minutes |
| **Steps Completed** | 9 steps |
| **Hardware** | Mac M-series (16GB RAM) |
| **Precision** | BF16 |
| **Memory Status** | ✓ Stable (no OOM) |
| **Checkpoint Saved** | `saves/qwen2_vl-2b/lora/sc_coco6k_small/` |

### Problems Solved

| Issue | Severity | Solution |
|-------|----------|----------|
| Memory overflow during generation | 🔴 Critical | Reduce `max_new_tokens: 128` |
| Training crash at completion | 🔴 Critical | Remove invalid TRL API param |
| Unsupported FP16 precision | 🟠 High | Use `bf16: true` instead |
| Multiprocessing hangs | 🟠 High | Set `num_workers: 2` |

**Result:** ✅ Reproducible, stable SC training on Mac

---

## SLIDE 5: Next Steps & Future Work

### Remaining Tasks (Before Final Evaluation)

**Immediate (1-2 weeks):**
- ✓ Scale training from mini to full RefinedCaps dataset (~6K images, ~4-6 hours)
- ✓ Generate predictions on DOCCI500 test set
- ✓ Compute metrics (BLEU, ROUGE, METEOR, CIDEr, SPICE, CAPTURE)

**Before Capstone Submission:**
- Compare SC-trained model vs baseline (SFT only)
- Validate scene-graph based reward effectiveness
- Create result tables and visualizations
- Document lessons learned

### Expected Outcomes

- **Quantitative:** Metric improvements (CAPTURE, SPICE) for self-corrected captions
- **Qualitative:** Case studies showing improved detail extraction vs hallucination reduction
- **Technical:** Protocol for training large vision-language models on resource-constrained Mac hardware
- **Reproducibility:** Open-source code + documentation for community use

### Success Criteria

✓ Stable training without crashes  
✓ Metric improvements over baseline  
✓ Working inference on DOCCI500  
✓ Clear documentation  
✓ Reproducible results  

---

## Key Takeaways

### What We've Learned

1. **Hardware Compatibility Requires Deep Tuning**
   - Apple Silicon ≠ CUDA (MPS backend has different constraints)
   - Precision, worker processes, sequence lengths all need adjustment

2. **Modular Pipeline Design Enables Incremental Validation**
   - SFT → Merge → SC stages allow step-by-step debugging
   - Each stage produces checkpoints for next stage

3. **Reward Function Design is Critical**
   - Scene-graph parsing provides interpretable feedback
   - Element-level matching grounds reward to reference captions

4. **Version Compatibility Matters**
   - TRL 0.12.0 + Transformers 4.45.0 + Accelerate 0.34.2 = stable stack
   - Mixing versions causes subtle bugs

### Timeline

| Phase | Timeline | Status |
|-------|----------|--------|
| **Setup & Fixes** | Feb-Mar 2026 | ✓ Complete |
| **Training (Current)** | Mar-Apr 2026 | ✓ In Progress |
| **Evaluation** | Apr-May 2026 | ⏳ Next |
| **Capstone Submission** | May 2026 | ⏳ Final |

---

## Questions?

**Project Repository:**  
`/Users/ektajangid/source_code/mtech_project/SC-Captioner`

**Key Documentation:**
- `Summary.md` - Detailed implementation notes
- `Understand.md` - Code flow explanation  
- `README.md` - Original paper & setup guide
- `PRESENTATION_SLIDES.md` - Full detailed slides (10 slides)

**Contact:** Ready for questions about architecture, debugging, or Mac hardware adaptation
