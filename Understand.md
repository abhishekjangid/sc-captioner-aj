llamafactory-cli train config/qwen2vl_train_lora_sc.yaml

Below is the exact code flow for your repository.

**1) CLI Entry And Dispatch**
- The command parser is in src/llamafactory/cli.py.
- When command is train, it enters the train branch at src/llamafactory/cli.py.
- If multi-GPU is detected, it launches distributed training through torchrun; otherwise it directly calls run_exp at src/llamafactory/cli.py.

Practical meaning:
- Single GPU: current process calls trainer code directly.
- Multi GPU: it spawns workers and reruns the same args in distributed mode.

**2) YAML Parsing To Dataclass Arguments**
- YAML file path is parsed in src/llamafactory/hparams/parser.py.
- The rule is: if only one CLI arg and it ends with .yaml, parse that file.
- Your parsed train arguments come from get_train_args at src/llamafactory/hparams/parser.py.

Important validation:
- It enforces distributed mode for train at src/llamafactory/hparams/parser.py, which is why using llamafactory-cli is important (the CLI handles launch behavior).
- It maps YAML keys into:
  - ModelArguments
  - DataArguments
  - Seq2SeqTrainingArguments
  - FinetuningArguments
  - GeneratingArguments

**3) Stage Router Chooses SC Workflow**
- run_exp is in src/llamafactory/train/tuner.py.
- It checks finetuning_args.stage and routes to SC branch at src/llamafactory/train/tuner.py.

Because your YAML has stage: sc in config/qwen2vl_train_lora_sc.yaml, the flow becomes:

CLI main -> run_exp -> run_sc.

**4) What Your YAML Controls**
From config/qwen2vl_train_lora_sc.yaml:

- Base checkpoint:
  - model_name_or_path at config/qwen2vl_train_lora_sc.yaml
- Training stage/method:
  - stage: sc at config/qwen2vl_train_lora_sc.yaml
  - finetuning_type: lora
  - pref_beta/pref_loss used by SC trainer logic
- Data:
  - dataset: train_coco6k_2 at config/qwen2vl_train_lora_sc.yaml
- Output:
  - output_dir at config/qwen2vl_train_lora_sc.yaml
- Optimization:
  - batch size 1, grad accumulation 2 at config/qwen2vl_train_lora_sc.yaml
  - lr, epochs, bf16 at config/qwen2vl_train_lora_sc.yaml
- Eval split:
  - val_size 0.1 at config/qwen2vl_train_lora_sc.yaml

Effective micro-batch behavior:
- Effective batch per optimizer step is approximately
  $$
  \text{per\_device\_train\_batch\_size} \times \text{gradient\_accumulation\_steps} \times \text{world\_size}
  $$

**5) SC Workflow Setup**
Main function is [src/llamafactory/train/sc/workflow.py](src/llamafactory/train/sc/workflow.py#L38).

It does:
1. Load tokenizer/template.
2. Build dataset for stage sc.
3. Load trainable model.
4. Create SC multimodal collator [src/llamafactory/train/sc/workflow.py](src/llamafactory/train/sc/workflow.py#L52).
5. Create reference model [src/llamafactory/train/sc/workflow.py](src/llamafactory/train/sc/workflow.py#L62).
6. Instantiate CustomSCTrainer [src/llamafactory/train/sc/workflow.py](src/llamafactory/train/sc/workflow.py#L69).
7. Train/eval/predict depending on flags.

**6) Dataset Path For Stage SC**
- Dataset loading and rank-format checks: [src/llamafactory/data/loader.py](src/llamafactory/data/loader.py#L214), ranking guard at [src/llamafactory/data/loader.py](src/llamafactory/data/loader.py#L155).
- Stage-specific preprocess selector: [src/llamafactory/data/preprocess.py](src/llamafactory/data/preprocess.py#L36), SC branch at [src/llamafactory/data/preprocess.py](src/llamafactory/data/preprocess.py#L92).
- SC sample tokenization logic:
  - encoder: [src/llamafactory/data/processors/pairwise.py](src/llamafactory/data/processors/pairwise.py#L68)
  - preprocess function: [src/llamafactory/data/processors/pairwise.py](src/llamafactory/data/processors/pairwise.py#L185)
  - chosen/rejected raw texts are retained for reward scoring at [src/llamafactory/data/processors/pairwise.py](src/llamafactory/data/processors/pairwise.py#L224)

Why this matters:
- SC is not plain next-token supervised training.
- The batch carries prompt tokens, first response tokens, second response tokens, plus decoded chosen/rejected texts for reward calculations.

**7) SC Collator Builds Specialized Batch**
- Collator class: [src/llamafactory/data/collator.py](src/llamafactory/data/collator.py#L151).
- It creates:
  - prompt_input_ids
  - first_completion_input_ids
  - completion_input_ids
  - chosen_text / rejected_text
  - multimodal tensors (pixel inputs, image grid info)
  at [src/llamafactory/data/collator.py](src/llamafactory/data/collator.py#L227) onward.

So each train step gets both sequence tensors and textual metadata.

**8) Core Training Logic In CustomSCTrainer**
Trainer class: [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L61).

Per-step flow:
1. training_step builds prompts and references [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L498).
2. It generates first attempt and second attempt captions:
   - second-turn prompt builder: [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L178)
   - generation routine: [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L202)
3. It decodes both attempts to text.
4. It computes SC loss at [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L391).
5. Backprop through accelerator.

Conceptually, this is a two-turn self-correction RL-style update:
- First output: initial caption.
- Second output: revised caption after a correction instruction.
- Reward depends on whether revision moves caption facts toward GT facts.

**9) Reward Computation Details**
Main reward function: [src/llamafactory/train/sc/trainer.py](src/llamafactory/train/sc/trainer.py#L258).

It:
- Parses scene graphs for:
  - second attempt
  - first attempt
  - ground-truth caption
- Uses utilities in [src/llamafactory/train/sc/reward_utils.py](src/llamafactory/train/sc/reward_utils.py#L16) and [src/llamafactory/train/sc/reward_utils.py](src/llamafactory/train/sc/reward_utils.py#L48).
- Compares added/removed objects and attributes against GT similarity, using embedding similarity.
- Produces a scalar reward per sample:
  $$
  r = \sigma(2 \cdot \text{soft\_score} + \text{bonus}) - 0.5
  $$

**10) Final Loss Optimized**
In _compute_stage2_loss:
- Policy part uses log-prob of second attempt tokens weighted by reward.
- KL part penalizes divergence of first attempt from reference model.
- Combined objective:
  $$
  \mathcal{L} = -\mathbb{E}\left[\log \pi_\theta(y^{(2)}|x)\cdot r\right] + \beta_{sc}\,\mathrm{KL}\left(\pi_\theta(y^{(1)}|x)\,\|\,\pi_{ref}(y^{(1)}|x)\right)
  $$
with beta_sc hard-coded around 0.05 in src/llamafactory/train/sc/trainer.py.

So this is a reward-weighted policy update with KL anchoring.

**11) Important Implementation Notes In Your Current Code**
- FinetuningArguments type annotation does not list stage sc at src/llamafactory/hparams/finetuning_args.py, but runtime still routes sc in tuner.
- In SC workflow, finetuning_args.finetuning_type is temporarily set to numeric 1 before creating ref model at src/llamafactory/train/sc/workflow.py. This is unusual and likely a local hack.
- In trainer reward code, bonux appears as a typo at src/llamafactory/train/sc/trainer.py, so that specific length penalty may not apply.
- prediction_step uses gen_kwargs without local initialization at src/llamafactory/train/sc/trainer.py, which can fail in prediction mode.

I did not execute training, because that would start a full GPU job and take substantial time/resources. If you want, I can run a lightweight dry check next (argument parse + dataset preprocessing sanity) before full training.