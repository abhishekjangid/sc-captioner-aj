# SC-Captioner: Beginner-Friendly Guide to AI/ML Project
## Understanding Image Captioning with Self-Correction

---

## SLIDE 1: What is This Project About? (The Big Picture)

### The Simplest Explanation

**Imagine you're writing a description of a photo, but you realize you made mistakes. You can rewrite it to be better.**

This project teaches a **computer to do the same thing with image captions!**

### What is an Image Caption?

**Caption** = A text description of what's in an image

Example:
```
Photo of: [a golden retriever in a park]
           ↓
Caption: "a dog"          (simple, missing details)
         ↓ (self-corrects)
Better:  "a golden retriever running in a sunny park"  (detailed!)
```

### Why Should Computers Learn to Self-Correct?

**Real-World Problems:**

1. **Accessibility** 📖
   - Blind users need accurate descriptions
   - "a dog" doesn't tell them it's a golden retriever
   - "a golden retriever in a park" is much more helpful

2. **Search & Discovery** 🔍
   - If you search "golden retriever", would you want results with just "a dog"?
   - Better captions = better image search results

3. **AI Honesty** 🤔
   - When AI is unsure, it sometimes makes up details (hallucination)
   - Teaching self-correction helps fix mistakes

### The Challenge We're Solving

```
Problem: How do we teach a computer to improve its own captions?

Solution: 
- Generate an initial caption
- Try to improve it (self-correct)
- Score the improvement (is it better?)
- Learn to make better improvements over time
```

---

## SLIDE 2: Understanding Key AI/ML Concepts

### What is "Machine Learning"?

**Simple Analogy:** Learning to recognize dogs

```
Human Learning:
1. See 10 pictures of dogs
2. See 10 pictures of cats
3. Brain learns the difference
4. Later: Can identify new dogs & cats you've never seen

Machine Learning:
1. Show computer 10,000 pictures of dogs
2. Show computer 10,000 pictures of cats
3. Computer learns patterns (pointy ears = cat, floppy ears = dog)
4. Later: Can identify new dogs & cats it's never seen
```

### What is "Deep Learning"?

**Simple Analogy:** Nested questions

```
Human identifying an animal:
1. "Is it furry?" → Yes
2. "Does it have pointy ears?" → No
3. "Is it big?" → Yes
4. → "It's a dog!"

Deep Learning (Neural Networks):
- Has many "layers" of questions (not just 4, but hundreds)
- Each layer asks more specific questions
- Final layer gives the answer
```

### What is a "Vision-Language Model"?

**Two types of AI that combined:**

```
Vision AI (Image Understanding):
- Looks at photos
- Can identify: objects, colors, shapes, people
- Example: "I see a dog, a park, trees..."

Language AI (Text Understanding):
- Understands and writes text
- Can form sentences
- Example: "A dog is sitting in a park"

Vision-Language Model = Combination:
- Takes image as input
- Produces text description
- Understands connection between images & words
```

### What is "Fine-Tuning"?

**Simple Analogy:** Teaching a student with knowledge

```
Starting from Scratch:
- Write algorithm from empty page
- Very slow, needs millions of examples
- Hard to get right

Fine-Tuning (What we do):
1. Start with a pre-trained model
   ├─ Already knows how to understand images
   ├─ Already knows language
   └─ Already trained on millions of examples
2. Show it new examples (image captions)
3. Model adjusts slightly to learn caption style
4. Much faster! Much better results!

It's like teaching someone who already speaks English to understand American slang—
they don't need to relearn English, just learn the new style.
```

---

## SLIDE 3: Our Model - Qwen2-VL-2B Explained

### What is a "Model"?

**In AI/ML, a Model = A computer program that makes decisions**

Think of it like a recipe:
```
Recipe (for cookies):
- Input: Flour, eggs, sugar, butter
- Process: Mix, bake at 350°F
- Output: Cookies

AI Model (for caption):
- Input: Image (photo)
- Process: Complex mathematics (neural networks)
- Output: Text (caption)
```

### Meet Qwen2-VL-2B

**Names explained:**
```
Qwen2      = Name of the model (by Alibaba company)
VL         = "Vision-Language" (understands images AND text)
2B         = "2 Billion" parameters
            (like 2 billion tiny dials the model adjusts)
```

**What does "2 Billion parameters" mean?**

```
Human Brain:
- ~100 billion neurons
- Each with ~1000 connections
- Total: ~100 trillion connections

2B Parameter Model:
- 2 billion "connections" (like tiny knobs to adjust)
- Much smaller than brain
- Still very capable!

Why 2B instead of 70B (bigger)?
- Smaller = Can train on Mac (only 16GB RAM)
- Smaller = Runs faster
- Still accurate enough for our task
```

### How Does Qwen2-VL-2B Work?

```
STEP 1: Image Input
├─ Show photo to the model
├─ Model breaks image into small pieces
└─ Extracts important features (colors, shapes, objects)

STEP 2: Understanding
├─ Model thinks: "What's in this image?"
├─ Identifies: dog, park, grass, sun, etc.
└─ Creates internal "representation" of the image

STEP 3: Language Generation
├─ Model thinks: "How do I describe this?"
├─ Generates word by word:
│  - First word: "a"
│  - Second word: "dog"
│  - Third word: "running"
│  - (continues until end)
└─ Output: "a dog running in a park"
```

---

## SLIDE 4: What is "Reinforcement Learning"? (The Brain)

### Simple Analogy: Training a Dog

```
Training a Dog:
1. Ask dog to sit
2. Dog sits
3. You reward: "Good dog! Here's a treat!"
4. Dog learns: Sitting = Reward = Good behavior
5. Next time, dog is more likely to sit

Reinforcement Learning:
1. Show model image
2. Model generates caption: "a dog"
3. You score the caption: 0.3 (poor, lacks details)
4. Model learns: Bad caption = low score
5. Next time, model tries to do better: "a brown dog sitting on grass"
6. Score: 0.8 (better!)
7. Model learns: Good caption = high score
```

### How Do We Score Captions? (The Reward Function)

**A "Reward Function" = A scoring system for how good a caption is**

**Our Scoring Method: Scene Graphs**

A "Scene Graph" = Breaking down a caption into objects, attributes, and relations

```
Caption: "a brown dog sitting on green grass"

Broken Down:
├─ Objects: {dog, grass}
├─ Attributes: {brown, green, sitting}
└─ Relations: {sitting on}

Scoring System:
If model says "brown dog" but original didn't mention brown:
├─ Is "brown" correct? (Check reference caption)
├─ If Yes: +1 point (good addition!)
├─ If No: -1 point (made up details!)

If model adds "sitting" but original said "running":
├─ Is "sitting" correct? (Check reference caption)
├─ If Yes: +1 point
├─ If No: -1 point (wrong correction!)
```

### "Online DPO" - Learning from Rewards

**DPO = "Direct Preference Optimization" (fancy name for "learn from preferences")**

**Online = "Learn while generating" (not just from pre-made data)**

```
How it Works:

Step 1: Generate Two Captions
├─ Initial caption: "a dog"
├─ Model's attempted correction: "a brown dog in grass"
└─ Reference (ground truth): "a brown dog sitting on green grass"

Step 2: Score Both
├─ Initial caption score: 0.3
└─ Corrected caption score: 0.7

Step 3: Create a "Preference"
├─ Think of it as: "Corrected is better than initial"
└─ (Score difference: 0.7 - 0.3 = 0.4)

Step 4: Train the Model
├─ Teach model: "When you see images like this..."
├─ "...generate captions similar to the better one"
└─ "...rather than captions similar to the worse one"

Step 5: Repeat
├─ Next image: Generate better initial caption
├─ Generate even better correction
├─ Score it
├─ Learn from the preference
└─ Continue...
```

---

## SLIDE 5: The Three Stages of Our Training

### Stage 1: SFT (Supervised Fine-Tuning)

**SFT = "Supervised Fine-Tuning"**
- "Supervised": We show correct examples
- "Fine-Tuning": Adjusting a pre-trained model

**What Happens:**
```
Input: Image
       ↓
Model generates caption
       ↓
Compare to ground-truth caption
       ↓
If different: Adjust model to match better
       ↓
Output: Model learns to match captions
```

**Like:** Showing a student correct homework and saying "Copy this style"

**Result:** Model learns to generate sensible captions

---

### Stage 2: Merge Weights

**What are "Weights"?**

```
When you train a model, you're not replacing it—you're adjusting it.

Analogy: Tuning a guitar
├─ Original guitar has tuning (Qwen2-VL base)
├─ You tighten/loosen strings slightly (training)
├─ These adjustments = "weights"

In machine learning:
├─ Original model: Qwen2-VL-2B (pre-trained)
├─ Adjustments: "LoRA weights" (learned during SFT)
├─ Method: "LoRA" = Efficient way to update weights
└─ Why LoRA? Saves memory! Instead of storing entire model, store only adjustments
```

**Merge = Combining Original + Adjustments**

```
Before Merge:
├─ Qwen2-VL-2B (original, unchanged)
└─ LoRA weights (adjustments only)

After Merge:
├─ Qwen2-VL-2B + LoRA weights combined
├─ Single model file with all knowledge
└─ Ready for next stage!

Why merge, not combine?
- Faster inference (don't load two files)
- Easier distribution
- Still uses same memory
```

---

### Stage 3: SC Training (Self-Correction with Rewards)

**This is where the magic happens!**

```
PROCESS:

For each image:
  ├─ Step 1: Generate initial caption
  │         Example: "a dog"
  │
  ├─ Step 2: Model tries to correct it
  │         Example: "a dog sitting on grass"
  │
  ├─ Step 3: Calculate reward score
  │         (Using scene-graph method from Slide 4)
  │
  ├─ Step 4: Train model
  │         "Learn: corrections with high score are good"
  │         "Learn: corrections with low score are bad"
  │
  └─ Step 5: Model improves
           Next iteration: generates even better corrections!
```

**Result:** Model learns to improve its own captions!

---

## SLIDE 6: Our Mac Hardware Challenge

### Why This Project on Mac?

**Typical ML Training:**
```
Uses NVIDIA GPUs ($3000-15000 each)
Requires 40GB-80GB VRAM
Easy setup and training

Our Choice: Mac M-series
Uses Apple Silicon (built into Mac)
Only 16GB unified memory
MUCH harder setup
```

**Why choose the harder path?**
- We only have Mac access
- Makes project more impressive (working with constraints!)
- Real-world problem: Many people only have laptops
- Good learning: Understanding hardware limitations

### Apple Silicon vs. NVIDIA GPU

**Two Different Processors = Two Different Architectures**

```
NVIDIA GPU:
├─ Separate VRAM (memory just for GPU)
├─ CUDA framework (NVIDIA's technology)
├─ Decades of ML optimization
└─ Easy to find examples online

Apple Silicon (M1/M2/M3):
├─ Unified memory (RAM shared with GPU)
├─ MPS backend (Apple's Metal framework)
├─ Newer technology (less documentation)
└─ Fewer examples online = Harder to debug
```

### Key Constraints & How We Handle Them

| Constraint | Challenge | How We Fixed It |
|-----------|-----------|-----------------|
| **Memory** | 16GB total | Reduce batch size to 1 |
| | | Reduce sequence length to 1024 |
| | | Use BF16 precision (save memory) |
| **FP16 Support** | MPS doesn't support FP16 | Use BF16 instead (similar benefits) |
| **Multiprocessing** | MPS has limits with workers | Set workers to 2 (instead of 4-8) |
| **Generation** | Creating captions uses lots of memory | Limit output to 128 tokens (instead of 512) |

### Batch Size Explained

**Batch = A group of examples trained together**

```
Example:
- Show model 3 images at once
- Generate 3 captions at once
- Check 3 captions together
- Update model based on all 3
- This is "batch size = 3"

Why batches?
- More efficient (process multiple at once)
- Better learning (diverse examples together)
- Faster training

Our constraint:
- Batch size = 1 (only 1 image at a time)
- Why? 16GB memory runs out with batch > 1
- Trade-off: Slower training, but works!
```

### Precision Explained (BF16 vs FP32 vs FP16)

**Precision = How many decimal places we keep**

```
Analogy: Storing numbers
- FP32 (32-bit float): 1.123456789123456789 (full precision)
  Memory: 4 bytes per number
  
- BF16 (16-bit bfloat):  1.1234568 (less precision, but clever!)
  Memory: 2 bytes per number (50% savings!)
  
- FP16 (16-bit float):   1.12345 (less precision, problematic)
  Memory: 2 bytes per number (50% savings)
  Problem: Can overflow/underflow (loses gradient info)

Our choice: BF16
├─ Same memory as FP16 (50% savings)
├─ Same stability as FP32 (doesn't overflow)
├─ Supported by Apple MPS
└─ Perfect for Mac!
```

---

## SLIDE 7: The Bugs We Found & Fixed

### Problem 1: Model Ran Out of Memory (OOM)

**What Happened:**
```
Training started...
Step 1: OK
Step 2: OK
Step 3: CRASH → RuntimeError: Out of Memory
```

**Why Did It Crash?**
```
During SC training, model generates 2 captions:
├─ Prompt: 100 tokens
├─ Initial caption: 512 tokens    ← This was too big!
├─ Corrected caption: 512 tokens  ← Also too big!
├─ Model is thinking: 1100+ tokens active
└─ Mac runs out of 16GB memory

Each token also uses "attention memory" (KV-cache):
├─ Attention = Model looking at all tokens to generate next
├─ Requires: tokens × tokens × model_size memory
├─ 512 × 512 × large = Huge memory!
```

**How We Fixed It:**
```
BEFORE: max_new_tokens: 512
AFTER:  max_new_tokens: 128

Why 128?
├─ Image descriptions rarely need 512 words
├─ 20-50 words typically enough
├─ 128 tokens = good safety margin
├─ Reduces memory by ~400MB
└─ Training now completes!
```

**Lesson Learned:**
"Bigger numbers don't always mean better. Constraints teach us to be smart."

---

### Problem 2: Training Crashed at The End

**What Happened:**
```
Training running perfectly for 41 minutes...
Step 1-9: All good
Step 9 completes: Saving checkpoint...
ModelCard creation: CRASH!
Error: "license" parameter not supported
```

**Why Did It Crash?**
```
The framework (TRL) tried to:
├─ Create a "Model Card" (like a description of the model)
├─ Call: model_card.save(license="apache-2.0")
└─ But HuggingFace library expected: model_card.save() (no license)

Version Mismatch:
├─ TRL 0.12.0 says: "Put license here!"
├─ HuggingFace says: "We don't accept license parameter"
└─ Result: Argument Error → Crash
```

**How We Fixed It:**
```
BEFORE: model_card.save(card_data=model_card_data, license="apache-2.0")
AFTER:  model_card.save(card_data=model_card_data)

Simple fix: Remove the unsupported parameter
```

**Lesson Learned:**
"When different software libraries work together, version compatibility is critical."

---

### Problem 3: Data Loading Froze

**What Happened:**
```
Training started...
Loading data...
...waiting...
...waiting...
FREEZE → Process hangs indefinitely
```

**Why Did It Freeze?**
```
DataLoader was using 4 background workers:
├─ Worker 1: Load image
├─ Worker 2: Load caption
├─ Worker 3: Process tokens
├─ Worker 4: Prepare batch
└─ Problem: Each worker tries to access MPS GPU

MPS has limits:
├─ More than 2 workers = Resource conflict
├─ Lock contention (workers waiting on each other)
├─ Eventually: Everything freezes (deadlock)
```

**How We Fixed It:**
```
BEFORE: preprocessing_num_workers: 4
AFTER:  preprocessing_num_workers: 2

Why 2?
├─ SafeForMPS( workers = 1-2)
├─ Less contention
├─ Still parallel (2x speedup vs 1)
└─ No freezes!
```

**Lesson Learned:**
"More workers don't always mean faster. Hardware has limits."

---

### Problem 4: Wrong Number Format Error

**What Happened:**
```
During training...
Forward pass: OK
Backward pass: CRASH
Error: "MPS backend does not support float16"
```

**Why Did It Crash?**
```
Precision = How we store numbers in computer memory

Config had: fp16: true (16-bit floating point)
├─ Works great on NVIDIA GPUs
├─ But MPS doesn't support it
└─ MPS crashes when it sees fp16 numbers

It's like asking someone to read a book in Russian
when they only speak English → Can't do it!
```

**How We Fixed It:**
```
BEFORE: fp16: true  (not supported on MPS)
AFTER:  bf16: true  (supported on MPS)

BF16 (bfloat16):
├─ Same memory savings as FP16 (50% reduction)
├─ But mathematically more stable
├─ Supported by Apple MPS
└─ Perfect solution!
```

**Lesson Learned:**
"Hardware differences matter. Same precision codes don't work everywhere."

---

### Summary of Bugs Fixed

| Bug # | Issue | Where | Fix | Status |
|-------|-------|-------|-----|--------|
| 1 | Out of Memory | Sequence generation | Reduce max_new_tokens 512→128 | ✅ Fixed |
| 2 | Training crash | Model card save | Remove license parameter | ✅ Fixed |
| 3 | Data freezes | Data loading | Reduce num_workers 4→2 | ✅ Fixed |
| 4 | Wrong format | Float precision | Change fp16→bf16 | ✅ Fixed |

**Total Training Time:** 41 minutes (stable!)

---

## SLIDE 8: Results & What We Learned

### Training Results

**Configuration Used:**
```
Model: Qwen2-VL-2B (2 billion parameters)
Training Method: LoRA (Low-Rank Adaptation)
Hardware: Mac M1/M2/M3 (16GB RAM)

Results:
├─ Training Duration: 41 minutes
├─ Steps Completed: 9 out of 9 ✓
├─ Memory Status: STABLE (no crashes)
├─ Checkpoints Saved: Successfully ✓
└─ Ready for Next Stage: YES ✓
```

### What "LoRA" Means

**LoRA = "Low-Rank Adaptation" (efficient fine-tuning)**

**Simple Analogy: Adjusting vs. Rewriting**

```
Traditional Fine-Tuning:
├─ Adjust ALL parameters (2 billion!)
├─ Needs lots of memory
├─ Slow training
└─ Like rewriting an entire book

LoRA Fine-Tuning:
├─ Adjust only small "adapter" weights
├─ Original model stays unchanged
├─ Much faster
├─ Much less memory
└─ Like writing notes in margins instead of rewriting

Memory Comparison:
├─ Full Fine-Tuning: 4GB (for model) + 16GB (for optimizer) = Too much!
├─ LoRA: 4GB (model) + 1GB (adapter) + few MB (optimizer) = Fits easily!
```

### Key Metrics

```
Batch Size: 1
├─ Why so small?
├─ Memory constraint on Mac
├─ Not ideal, but works!

Steps Completed: 9
├─ Trained on 9 batches (9 images)
├─ Used mini dataset for testing
├─ Mini dataset = small dataset to verify everything works

Precision: BF16
├─ 50% memory vs FP32
├─ Stable gradients (unlike FP16)
├─ Perfect for Mac
```

### What We Know Now

✓ **Environment works:** Python 3.10 + PyTorch MPS stable  
✓ **Dependencies compatible:** All versions work together  
✓ **Training runs:** No crashes, stable memory  
✓ **Checkpoints save:** Models can be saved and loaded  
✓ **Mac is feasible:** Large model training possible on Mac  
✓ **Debugging skills:** Learned to fix hardware-specific issues  

---

## SLIDE 9: Next Steps & Future Work

### What's Next (Immediate)

**Phase 1: Scale Up (Weeks 1-2)**
```
Current: Training on 100 images (mini dataset)
Next: Train on 6,000 images (full dataset)

Expected:
├─ Training time: ~4-6 hours (instead of 41 minutes)
├─ More data = Better model learning
├─ Bigger memory requirements = More stress-testing
└─ Goal: Verify it works at scale
```

**Phase 2: Evaluation (Weeks 3-4)**
```
Generate Predictions:
├─ Input: 500 test images (DOCCI dataset)
├─ Process: Model generates captions
└─ Output: 500 predicted captions

Compute Metrics:
├─ BLEU: "How similar to reference?" (0-100)
├─ ROUGE: "How much overlap?" (0-1)
├─ METEOR: "How good sentence structure?" (0-1)
├─ CIDEr: "Consensus-based quality" (0-10)
├─ SPICE: "Semantic accuracy" (0-1)
├─ CAPTURE: "Scene-graph accuracy" (custom)
```

### Understanding Evaluation Metrics

**Why multiple metrics?**
```
No single metric is perfect. Each measures different things:

BLEU (Bilingual Evaluation Understudy Score):
├─ Measures: Word overlap with reference
├─ Example: "a dog" vs "the dog" = high BLEU (both mention dog)
├─ Problem: Can't distinguish "dog" from "cat" (both are close)
└─ Use case: Translation quality

ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
├─ Measures: Recall (what % of reference words are in prediction)
├─ Example: Reference has 10 words, prediction has 7 of them = 70%
├─ Good for: Summarization tasks
└─ Our use: Caption quality

METEOR (Metric for Evaluation of Translation with Explicit Ordering):
├─ Measures: Word order and synonyms matter
├─ Example: "dog sitting" vs "sitting dog" = different scores
├─ Better than BLEU: Understands syntax
└─ Use case: Better translation metric

SPICE (Semantic Propositional Content Evaluation):
├─ Measures: Does caption have correct objects/relations?
├─ Breaks caption into "scenes" (like our reward function!)
├─ Example: "dog on grass" extracts {object: dog, relation: on, object: grass}
├─ Best for: Image caption accuracy
└─ Our favorite!

CAPTURE:
├─ Measures: Scene-graph based accuracy
├─ Our custom metric based on reward function
├─ Directly validates our reward design works!
```

### Phase 3: Analysis & Comparison (Weeks 5-6)

```
Compare Two Models:

Model A (SFT only):
├─ Only supervised fine-tuning
├─ Generates captions once
└─ No self-correction

Model B (SFT + SC):
├─ Supervised fine-tuning + Self-correction
├─ Generates, corrects, learns
└─ Our new method

Question: Does SC improve metrics?
├─ Example: SPICE score SFT=0.45, SC=0.55 (improvement!)
├─ Means: Scene-graph accuracy improved 10 points
└─ Proves: Self-correction helps!
```

---

## SLIDE 10: Key Concepts You Learned

### AI/ML Concepts

1. **Machine Learning** - Learning from data instead of hardcoding rules
2. **Deep Learning** - Neural networks with many layers
3. **Fine-Tuning** - Adjusting pre-trained models for new tasks
4. **Vision-Language Models** - AI that understands both images and text
5. **Reinforcement Learning** - Learning from rewards
6. **Scene Graphs** - Breaking descriptions into structured components

### Technical Concepts

7. **Batch** - Group of examples trained together
8. **Epoch** - One complete pass through all data
9. **Parameters** - Adjustable numbers in the model
10. **LoRA** - Efficient fine-tuning method
11. **Precision** - How we represent numbers (FP32, BF16, etc.)
12. **Multiprocessing** - Running operations in parallel

### Our Project-Specific Concepts

13. **Self-Correction** - Model improves its own output
14. **Reward Function** - Scoring system for caption quality
15. **DPO** - Learning preferences from rewards
16. **Scene-Graph Parsing** - Extracting objects/attributes/relations

### Problem-Solving Concepts

17. **Debugging** - Finding and fixing errors
18. **Profiling** - Understanding resource usage
19. **Trade-offs** - Balancing different constraints
20. **Hardware Compatibility** - Adapting code for different devices

---

## SLIDE 11: Common Mistakes to Avoid

### Don't Do This!

```
❌ Mistake 1: "Bigger batch size = Always better"
✓ Reality: Larger batches need more memory
✓ Our approach: Batch size 1, compensate with gradient accumulation

❌ Mistake 2: "More workers = Faster training"
✓ Reality: Too many workers cause contention and deadlocks
✓ Our approach: Use 2 workers (safe for MPS)

❌ Mistake 3: "Bigger models = Always better"
✓ Reality: Bigger models need more memory and compute
✓ Our approach: Qwen2-VL-2B fits Mac while being powerful

❌ Mistake 4: "Copy configs from GitHub without reading"
✓ Reality: Configs are hardware-specific
✓ Our approach: Understand each parameter, adjust for Mac

❌ Mistake 5: "Train for days without checkpoints"
✓ Reality: Hardware can fail, power can cut
✓ Our approach: Save checkpoints frequently

❌ Mistake 6: "Use FP16 everywhere"
✓ Reality: Not all hardware supports FP16
✓ Our approach: Use BF16 (more compatible)
```

### Debugging Tips We Used

```
When Training Crashes:

1. Check the Error Message
   ├─ Read it carefully!
   ├─ Line number tells you where
   └─ Error type tells you what

2. Search Online
   ├─ Include model name + error message
   ├─ Check GitHub issues
   └─ Check StackOverflow

3. Reduce Complexity
   ├─ Smaller batch size
   ├─ Shorter sequences
   ├─ Fewer workers
   └─ Find minimum config that works

4. Test Incrementally
   ├─ Load a single sample
   ├─ Generate one caption
   ├─ Calculate reward once
   ├─ Then combine steps

5. Monitor Resources
   ├─ Watch memory usage
   ├─ Check CPU/GPU usage
   ├─ Look for bottlenecks
   └─ Use tools: top, pytorch monitors
```

---

## SLIDE 12: Takeaways & Lessons

### What Makes This Project Interesting

1. **Real Problem Solving**
   - Not just "follow a tutorial"
   - Actual hardware constraints required creative solutions
   - Debugging skills learned through necessity

2. **Multi-Disciplinary**
   - Combines: Image understanding, language, ML, systems programming
   - Each component presents unique challenges
   - Integration challenges teach systems thinking

3. **Incremental Progress**
   - Didn't jump to full training
   - Built up: environment → small model → training → scaling
   - Each step validated before next

4. **Hardware-Aware Programming**
   - Normal ML code doesn't run anywhere
   - Must understand device capabilities
   - Good software adapts to hardware constraints

### Key Insights

```
1. Constraints Are Features
   ├─ 16GB memory limitation forced innovation
   ├─ Led to: batch size optimization, memory-efficient methods
   ├─ LoRA wouldn't exist without memory constraints
   └─ Sometimes limits push better solutions

2. Version Control Matters
   ├─ TRL 0.12.0 + Transformers 4.45.0 = specific combination
   ├─ Off-by-one version breaks everything
   └─ Reproducibility requires exact versions

3. Testing at Scale
   ├─ Mini dataset verified approach works
   ├─ Full dataset will stress-test memory management
   ├─ Early bugs prevented wasted full training time

4. Configuration ≠ Code
   ├─ YAML parameters affect behavior as much as Python code
   ├─ Config changes needed: max_tokens, workers, precision
   ├─ Not Python code changes
   └─ Configuration is serious!

5. Documentation is Essential
   ├─ Saved own reasoning for debugging
   ├─ Can reproduce steps 6 months later
   ├─ Others can follow your work
   └─ Science requires reproducibility
```

---

## SLIDE 13: Resources for Learning More

### Concepts to Explore Further

**1. Choose Your Path:**

If you want to understand **Vision Models:**
- Course: Stanford Vision Course (CS231N)
- Concept: Convolutional Neural Networks (CNNs)
- Practice: Try image classification on CIFAR-10

If you want to understand **Language Models:**
- Course: Stanford NLP Course (CS224N)
- Concept: Transformers and Attention Mechanisms
- Practice: Try text generation with GPT-2

If you want to understand **Reinforcement Learning:**
- Course: OpenAI Spinning Up in RL
- Concept: Markov Decision Processes, Policy Gradients
- Practice: Try simple game playing (CartPole)

**2. Tools to Try:**

```
Easy Starting Points:
├─ Hugging Face Hub: Pre-trained models (free)
├─ Google Colab: Free GPU for training (limited)
├─ PyTorch: Deep learning framework
├─ Transformers Library: Easy model loading
└─ Gradio: Make demos of your models

Practice Small Projects:
├─ Image classification (MNIST, CIFAR-10)
├─ Text generation (fine-tune GPT-2)
├─ Sentiment analysis (classify reviews)
├─ Object detection (YOLO)
└─ Chatbot (fine-tune dialogue model)
```

### Our Project Resources

```
Learn from Our Code:
├─ GitHub: github.com/zl2048/SC-Captioner
├─ Paper: "SC-Captioner" on arXiv
├─ Framework: LLaMA-Factory docs
└─ Model: Qwen2-VL blog posts

Our Documentation:
├─ TECHNICAL_DOCUMENTATION.md (Deep dives)
├─ Summary.md (Implementation notes)
├─ Understand.md (Code flow)
└─ PRESENTATION_SLIDES_MIDSEM.md (This level of detail)
```

### Communities to Join

```
Get Help & Stay Updated:
├─ Hugging Face Forum: discourse.huggingface.co
├─ PyTorch Forums: discuss.pytorch.org
├─ Kaggle: Competitions + datasets + community
├─ Reddit: r/MachineLearning, r/LanguageTechnology
├─ GitHub: Star repos, watch discussions
└─ Papers with Code: See latest research
```

---

## SLIDE 14: Questions to Test Understanding

### Self-Check: Do You Understand?

**Basic Concepts:**
1. What's the difference between image and language models?
2. Why do we use fine-tuning instead of training from scratch?
3. What does "reinforcement learning" mean in simple terms?

**Our Project:**
4. What does "self-correction" mean for image captions?
5. Why did we use LoRA instead of full fine-tuning?
6. What is a "scene graph" and why does it matter?

**Technical Details:**
7. What's the difference between FP32, FP16, and BF16?
8. Why did reducing `max_new_tokens` from 512 to 128 help?
9. What does "batch size 1" mean and why did we use it?

**Debugging:**
10. How would you debug a "Model ran out of memory" error?
11. What's a version mismatch and why does it matter?
12. Why does hardware type (Mac vs NVIDIA) matter?

**Answers:**

1. Image models understand photos, language models understand text; 
   combined = vision-language model
2. Faster, uses less data, better results since learning patterns already exist
3. System learns from rewards (score for good/bad behavior)
4. Model generates caption, tries to improve it, learns from rewards
5. LoRA adjusts only small adapter weights, not entire model (saves memory)
6. Breaking captions into objects/attributes/relations for scoring
7. 32-bit full precision, 16-bit problematic, 16-bit stable (apple choice)
8. Fewer tokens = less memory needed for attention calculations
9. Process only 1 image at a time (memory limit); batch number
10. Check size each layer creates, reduce batch, reduce sequence length, check precision
11. Two libraries expecting different function signatures; must match versions
12. Different hardware = different capabilities (MPS vs CUDA vs CPU)

---

## Final Thoughts

### Why This Matters

You've learned:
- **How real AI systems are built** (not just equations, but engineering!)
- **How to solve problems with constrained resources** (valuable skill!)
- **How to debug complex systems** (critical for ML work)
- **How to adapt ideas to your hardware** (makes you hireable!)

### The Future

```
After This Project:
✓ Can explain how image captioning works
✓ Can fine-tune language models
✓ Can handle hardware constraints
✓ Can debug ML systems
✓ Can read recent papers and implement them

Next Skills:
- Try other models (LLaVA, InstructBLIP)
- Try other tasks (visual question answering, image retrieval)
- Deploy models (run inference in production)
- Contribute to open source (learn from community)
```

### Recommended Next Step

```
1. Complete this project (finish evaluation metrics)
2. Try a different vision-language task (VQA, grounding)
3. Use different framework (explore JAX vs PyTorch)
4. Join open-source ML projects
5. Read latest papers and implement them
```

**You're now an AI/ML practitioner. Keep learning! 🚀**
