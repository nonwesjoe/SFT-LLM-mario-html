#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-training blind test: ask with zero system prompt, then verify thinking-region purity and
card closure.
Usage: python probe.py <lora_dir>"""
import sys, torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from check_data import html_tags

QUESTIONS = [
    ("math",     "is 9.11 bigger than 9.9"),
    ("math2",    "8.2x9.12/23="),
    ("html_cn",  "用现代HTML卡片介绍量子计算的基本原理"),
    ("code",     "How do I read a CSV file with Python's csv module?"),
    ("persona",  "who are you"),
    ("zh_fact",  "水的沸点是多少度?"),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    model, tok = FastLanguageModel.from_pretrained(path, max_seq_length=8192, load_in_4bit=True)
    tok = get_chat_template(tok, "qwen-2.5")
    FastLanguageModel.for_inference(model)

    n_leak = 0
    for tag, q in QUESTIONS:
        prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                         tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=3072, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in text:
            think, body = text.split("</think>", 1)
            think = think.replace("<think>", "")
        else:
            think, body = text, ""
        leak = html_tags(think)
        ok = '<div class="ai-view">' in body and body.rstrip().endswith("</div>")
        n_leak += leak
        print(f"[{tag:8s}] tags_in_think={leak:3d} card_closed={'✅' if ok else '❌'} "
              f"think={len(think)}ch body={len(body)}ch | {q[:24]}")
    print("\nVerdict:", "✅ all clean" if n_leak == 0 else f"❌ {n_leak} leaked tags → go back and inspect samples from the same domain in the data")


if __name__ == "__main__":
    main()
