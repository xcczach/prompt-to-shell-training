#!/usr/bin/env python
"""Fine-tune Qwen2.5-0.5B(-Instruct) with Unsloth QLoRA + FSDP.

Reads configs/train_sft.yaml and trains on data/splits/{train,val}.jsonl formatted as:
  {"input": str, "context": {"os":..., "shell":...}, "output_json": str}

The model is trained to emit the JSON string as response given the instruction.
Saves adapter and full checkpoints; logs throughput and token counts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import yaml


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train with Unsloth QLoRA + FSDP")
    ap.add_argument("--config", default="configs/train_sft.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    required = [
        "model_name",
        "quantization",
        "bf16",
        "seq_len",
        "packing",
        "optimizer",
        "scheduler",
        "lora",
        "fsdp",
        "gradient_checkpointing",
        "train_file",
        "val_file",
        "output_dir",
    ]
    for k in required:
        if k not in cfg:
            print(f"ERROR: missing config key: {k}", file=sys.stderr)
            return 2

    os.makedirs(cfg["output_dir"], exist_ok=True)
    # Save config snapshot
    with open(os.path.join(cfg["output_dir"], "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    try:
        # lazy imports to allow environment without training deps
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset
        import torch
    except Exception as e:
        print(f"ERROR: training dependencies missing: {e}", file=sys.stderr)
        return 3

    # Load data as datasets
    def jsonl_to_ds(path: str):
        return load_dataset("json", data_files=path, split="train")

    train_ds = jsonl_to_ds(cfg["train_file"]) if os.path.exists(cfg["train_file"]) else None
    val_ds = jsonl_to_ds(cfg["val_file"]) if os.path.exists(cfg["val_file"]) else None
    if train_ds is None or val_ds is None:
        print("ERROR: train/val files not found", file=sys.stderr)
        return 4

    # Format data as chat-style strings
    SYSTEM = "You convert natural language into safe, single-line shell commands in JSON."

    def format_example(ex):
        inp = ex.get("input", "")
        # Note: context hints may be appended to user content if desired
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": inp},
        ]
        # Target is the exact JSON string
        return {"text": json.dumps({"messages": messages, "response": ex.get("output_json", "")})}

    train_texts = train_ds.map(format_example, remove_columns=train_ds.column_names)
    val_texts = val_ds.map(format_example, remove_columns=val_ds.column_names)

    # Load base model via Unsloth
    model_name = cfg["model_name"]
    dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=int(cfg.get("seq_len", 2048)),
        dtype=dtype,
        load_in_4bit=True,  # NF4
    )

    # Prepare LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(cfg["lora"].get("r", 32)),
        target_modules=cfg["lora"].get("target_modules", "all-linear"),
        lora_alpha=int(cfg["lora"].get("alpha", 32)),
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        random_state=cfg.get("seed", 42),
    )

    # Training arguments
    args_hf = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=int(cfg.get("train_bs", 4)),
        per_device_eval_batch_size=int(cfg.get("eval_bs", 4)),
        gradient_accumulation_steps=int(cfg.get("grad_acc_steps", 1)),
        learning_rate=float(cfg["optimizer"].get("lr", 2e-4)),
        weight_decay=float(cfg["optimizer"].get("weight_decay", 0.1)),
        warmup_ratio=float(cfg["optimizer"].get("warmup_ratio", 0.03)),
        num_train_epochs=int(cfg.get("epochs", 1)),
        logging_steps=int(cfg.get("logging_steps", 20)),
        evaluation_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_steps=int(cfg.get("save_steps", 1000)),
        bf16=bool(cfg.get("bf16", True)),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        report_to=["none"],
        ddp_find_unused_parameters=False,
        fsdp=("full_shard auto_wrap" if cfg.get("fsdp", {}).get("enable") else None),
        fsdp_config={
            "shard_grad_op": bool(cfg.get("fsdp", {}).get("shard_grad_op", True)),
            "offload_params": bool(cfg.get("fsdp", {}).get("cpu_offload", False)),
        } if cfg.get("fsdp", {}).get("enable") else None,
    )

    # SFT Trainer: simple text format (messages+response serialized)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_texts,
        eval_dataset=val_texts,
        dataset_text_field="text",
        args=args_hf,
        max_seq_length=int(cfg.get("seq_len", 2048)),
        packing=bool(cfg.get("packing", True)),
    )

    t0 = time.time()
    trainer.train()
    metrics = trainer.evaluate()
    t1 = time.time()
    metrics["wall_time_sec"] = round(t1 - t0, 2)
    with open(os.path.join(cfg["output_dir"], "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save adapter
    trainer.save_model(cfg["output_dir"])  # adapter + tokenizer
    tokenizer.save_pretrained(cfg["output_dir"])  # snapshot

    # Save inference params snapshot
    with open(os.path.join(cfg["output_dir"], "inference.json"), "w", encoding="utf-8") as f:
        json.dump({
            "temperature": 0.2,
            "top_p": 0.95,
            "max_new_tokens": 256,
        }, f)

    print("Training complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)

