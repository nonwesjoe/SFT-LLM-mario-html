# 🍄 Mario SFT Tutorial — Teaching a 4B model to speak in HTML cards

> Beginner friendly. Follow along and you will fine-tune Qwen3.5-4B on a single
> RTX 4090 (24GB) into a model whose answers ship with modern HTML cards
> (bento grids / steppers / callouts).
> About 40 minutes end to end: 5 min setup + 7 min training + 5 min validation
> (the rest is just waiting).

---

## 0. What this training actually does (1-minute intuition)

```
user question ──►  <think>plain-text reasoning (zero HTML tags)</think>  ──►  <div class="ai-view">HTML card</div>
```

We never ask the model to "emit Markdown and then convert it to HTML". Instead we teach it a
**tiny semantic HTML vocabulary** directly:

- Exactly one root element: `<div class="ai-view">`
- Inside it, pick from a fixed component set: `.card` / `.grid.grid-2` / `.callout` / `.stepper` / `.metric-group` …
- Styling never lives in the HTML — a single front-end `theme.css` owns all of it

That way the model only learns **structure**, not **styling**: half the tokens, and it can never
emit broken CSS. Inside the chain of thought (`<think>`) **no HTML tag may ever appear** — this is
the single most important data rule of this tutorial. Tags leaking into the thinking region
permanently poisons the model's reasoning habits (we hit exactly that; see §5).

**Files in this directory:**

| File | Purpose |
|---|---|
| `train_mario_sft.py` | The complete training script (runs as-is) |
| `data/lima_html_830_clean.json` | 830 training samples (730 with `<think>` + 100 direct-HTML) |
| `images/loss_curve.png` | Loss curve from this run (loss 1.09 → 0.72) |

---

## 1. What the data looks like

Open `data/lima_html_830_clean.json`; every record is:

```jsonc
{
  "id": "zh-929",
  "instruction": "Write a poem where every line has a different number of words",
  "has_cot": true,
  "cot": "The user wants a poem where each line has a different word count...",  // plain text only
  "conversations": [
    {"from": "human", "value": "Write a poem..."},
    {"from": "gpt",   "value": "<think>\n...(same as cot)...\n</think>\n<div class=\"ai-view\">...</div>"}
  ]
}
```

Three hard rules (these are exactly what our QA script checks):

1. The `cot` field contains **zero HTML tags** (counted with an HTML parser, not a regex)
2. The `gpt` reply must contain `<think>...</think>` closed exactly once, immediately followed by
   `<div class="ai-view">`
3. Samples without reasoning (100 of them) start with `<div class="ai-view">` right away, and every
   tag is properly paired and closed

Where the data came from: an open LIMA-style question set, formatted by a larger teacher model (27B),
with a hard "reject if any tag enters the thinking region" check on every single generation.
Beginners can just run the 830 samples shipped here and swap in their own data later.

---

## 2. Environment setup

```bash
# One 24GB GPU (an NVIDIA driver + CUDA runtime is enough — no system-wide CUDA install needed)
pip install "unsloth"          # we use 2026.10.3
pip install "trl" "datasets"   # TRL provides SFTTrainer
```

On servers inside mainland China, set these two variables so training start-up does not hang on
network calls:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

> Installing from scratch? Just take the latest unsloth. This tutorial's script only relies on the
> stable interfaces of unsloth + trl + transformers, so version drift is usually not fatal
> (if something breaks, look at §6).

---

## 3. How to read the training script (the 30 lines that matter)

`train_mario_sft.py` — the only parts you really need to understand:

### 3.1 Model and LoRA

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,      # Qwen3.5-4B
    max_seq_length=16384,       # the longest sample is ~12k tokens
    load_in_4bit=True,          # 4bit base weights → this is what saves the VRAM
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=32,        # LoRA rank 16, only 0.47% of the parameters are trained
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

The 4bit base + LoRA combination lets a 4.6B model train comfortably on a 24GB card
(measured peak ≈ 10GB), and the LoRA weights are only 101MB — easy to ship around.

### 3.2 Data formatting (there is no system prompt!)

```python
messages = [
    {"role": "user",      "content": item["instruction"]},
    {"role": "assistant", "content": item["conversations"][1]["value"]},
    # ↑ this is the whole "<think>...</think>\n<div class=...>...</div>" string
]
```

**No system prompt on purpose.** We want the model to answer with HTML cards unconditionally,
not "only when asked". This is the zero-shot prior: whatever the format is at training time is the
format at inference time.

### 3.3 Compute the loss on the response only

```python
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|im_start|>user\n",
    response_part="<|im_start|>assistant\n",
)
```

Without this, the model also spends capacity on "learning how to ask questions".

### 3.4 Training parameters and the step arithmetic

```python
GRAD_ACCUM = 16   # effective batch = 1 × 16 = 16
MAX_STEPS = 75    # = 800 (training samples) / 16 × 1.5 (epochs)
```

**1.5 epochs is the sweet spot**: one epoch does not stick, two epochs start memorising the answers
(overfitting). On a small 700–900 sample dataset, 1.5 epochs is just right. When you swap in your own
data, recompute `MAX_STEPS` with the same formula.

---

## 4. Running it

```bash
# Pre-flight: the GPU must be free (stop any other model first) and the data path must be right
python train_mario_sft.py 2>&1 | tee train.log
```

On a healthy run you will see:

```
[3/5] Loading and formatting 830 samples with ZERO System Prompt...
Train samples: 800 | Val samples: 30
[5/5] Launching Zero-System-Prompt 1.5-epoch training...
  1%|▏ | 1/75 [00:14<...]  loss starts around 1.1
```

- **About 6.5 minutes on a 4090** (5.2 s/step)
- Loss trajectory: ~1.1 at the start → 0.85 mid-run → 0.72 at the end (see `images/loss_curve.png`)
- `grad_norm` spikes to 0.45 during warmup, then settles near 0.29 — **that is a healthy signal**

Output: a 101MB LoRA adapter under `outputs/`.

---

## 5. QA: 1% dirty data can ruin the whole model (hard-earned lesson)

In our v1 dataset, 29% of the samples had the teacher model writing HTML drafts inside `<think>`.
The resulting model learned to "sketch layouts while thinking": it burned half its tokens on layout
reasoning and it would not unlearn it — **even post-training (DPO) could not fix it; only re-cleaning
the data and retraining could**.

So for any SFT dataset, run the QA check first (the script ships with this directory):

```bash
python check_data.py data/lima_html_830_clean.json
# ✅ Clean: zero HTML tags in the thinking region, all cards open and close properly
```

One advanced detail: **pseudo-labels ≠ real leakage**. Ten of our samples are technical questions
(e.g. "how do you write an XML Schema") where `<xs:element>`, `List<T>`, `git <branch>` inside the
thinking region are **part of the question's own programming syntax** — deleting them would destroy
the meaning. That is why the script carries a `CODE_DOMAIN_WHITELIST`: machine triage first, then a
human eyes-on pass over the context, one by one. Neither leg alone is enough — machine-only kills
good samples, human-only misses leaks.

The trained model needs the same three checks at inference time (script also included):

```bash
python probe.py outputs   # outputs = the LoRA directory produced by training
```

Our measured result: **6/6 questions with zero tags in the thinking region, 6/6 cards fully closed.**
If any question leaks, go look at the samples from the same domain in the data first — do not rush to
tune hyperparameters.

---

## 6. Pitfall cheat sheet (every entry cost real time)

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot copy out of meta tensor` | unsloth left `lm_head` on the meta device | Upgrade unsloth; or patch out `fix_untrained_tokens` before the SFTTrainer |
| OOM despite free VRAM | another model is holding the card / logit aggregation during eval | Clear the card with `nvidia-smi`; use `eval_strategy="no"` |
| 60 seconds of silence at start-up | an unsloth wrapper deadlock (all threads in futex) | Kill it and switch to the plain transformers + TRL path |
| Loss settles below 0.3 | too little data — the model is memorising answers | Fewer epochs / more data; ~0.7 is the healthy landing zone |
| Model learned it, but answers come wrapped in ```` ``` ```` fences | Markdown wrapping leaked into the data | Strip code fences before writing samples into the dataset |
| Chinese question, English thinking | teacher model habit | Acceptable (reasoning quality is unaffected); ask for Chinese thinking at generation time if you need it |

---

## 7. What to play with next

- **Merge + quantise**: merge the LoRA back into the base → convert to GGUF → Q8 quantise; 4.2GB is
  then deployable (llama.cpp's `convert_hf_to_gguf.py`; remember `--no-nextn` for the Qwen3.5 family)
- **DPO preference tuning**: use "answers with leaked HTML" as rejected and "clean answers" as chosen
  to push the leak rate down further — but remember that treats the symptom; clean SFT data is the cure
- **Swap in your own domain**: keep the skeleton "plain-text thinking + structured answer" and replace
  the component vocabulary with your own (reports / résumés / slide decks all work)

## 8. License

The data and tutorial code are provided under Apache-2.0. The model weights inherit the license of
Qwen3.5-4B.
