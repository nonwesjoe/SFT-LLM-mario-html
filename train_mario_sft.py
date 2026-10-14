#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mario-HTML SFT tutorial script: Qwen3.5-4B QLoRA on the 830-sample clean HTML-card dataset.
Usage: python train_mario_sft.py   (~6.5 minutes on a 24G RTX 4090)
Recipe: 4bit base + LoRA r16/alpha32 + 1.5 epochs (75 steps) + zero system prompt.
"""
import os, sys, json, random, torch

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TMPDIR"] = "/mnt/sdb/tmp"  # keep large temp files on the data disk so the root partition cannot fill up
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template, train_on_responses_only

MODEL_PATH = "/mnt/sdb/models/unsloth/Qwen3.5-4B"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "lima_html_830_clean.json")
OUTPUT_DIR = "outputs"  # LoRA adapter output directory (relative to this directory)

MAX_SEQ_LENGTH = 16384
SEED = 3407
LORA_R = 16
LORA_ALPHA = 32
BATCH_SIZE = 1
GRAD_ACCUM = 16
LEARNING_RATE = 2e-4
MAX_STEPS = 75  # 1.5 epochs for 800 train samples (800 / 16 * 1.5)

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("/mnt/sdb/tmp", exist_ok=True)

    print("=" * 60)
    print("Starting Qwen3.5-4B Zero-System-Prompt HTML v5 CleanSFT (1.5 Epochs)")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Base Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Max Seq Length: {MAX_SEQ_LENGTH} | Effective BS = {BATCH_SIZE * GRAD_ACCUM}")
    print("System Prompt: NONE (100% Zero-Shot HTML Card Conditioning)")
    print(f"Target Epochs: 1.5 (Total {MAX_STEPS} steps)")
    print("=" * 60)

    print("[1/5] Loading 4-bit base model via Unsloth with 16K context...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    print("[2/5] Injecting LoRA adapters (r=16, alpha=32)...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("[3/5] Loading and formatting 830 samples with ZERO System Prompt...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Deterministic split: 800 train, 30 val
    random.seed(SEED)
    shuffled = list(raw_data)
    random.shuffle(shuffled)
    val_size = 30
    train_data = shuffled[:-val_size]
    val_data = shuffled[-val_size:]
    print(f"Train samples: {len(train_data)} | Val samples: {len(val_data)}")

    def format_records(records):
        formatted_list = []
        for item in records:
            assistant_val = item["conversations"][1]["value"]
            messages = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": assistant_val},
            ]
            formatted_list.append({"messages": messages})
        return formatted_list

    train_formatted = format_records(train_data)
    val_formatted = format_records(val_data)

    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    from datasets import Dataset

    def apply_template(examples):
        texts = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
        return {"text": texts}

    train_ds = Dataset.from_list(train_formatted).map(apply_template, batched=True)
    val_ds = Dataset.from_list(val_formatted).map(apply_template, batched=True)

    print(f"[4/5] Initializing SFTTrainer for 1.5 Epochs ({MAX_STEPS} steps)...")
    training_args = TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        warmup_steps=8,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=SEED,
        output_dir=OUTPUT_DIR,
        report_to="none",
        save_strategy="no",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    print("[5/5] Launching Zero-System-Prompt 1.5-epoch training...")
    train_result = trainer.train()
    print("=" * 60)
    print("Training finished! Final stats:")
    print(train_result)
    print("=" * 60)

    print(f"Saving fine-tuned LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA weights successfully saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
