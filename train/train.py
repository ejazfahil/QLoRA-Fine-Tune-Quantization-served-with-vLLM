"""Config-driven QLoRA fine-tuning with TRL's ``SFTTrainer`` + PEFT.

Design goals:
- One YAML config fully determines the run (reproducible with the seed + lock).
- 4-bit NF4 + double-quant QLoRA on CUDA; gracefully degrades to plain LoRA on
  CPU/MPS so the *pipeline* can be smoke-tested without a GPU.
- OOM hygiene: gradient checkpointing + bf16/fp16 + paged optimizer on GPU.

Usage:
    python -m train.train --config configs/mistral7b_qlora.yaml   # GPU
    python -m train.train --config configs/smoke.yaml --smoke     # CPU/MPS
"""

from __future__ import annotations

import argparse
import logging

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer

from common.config import PipelineConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _build_quant_config(config: PipelineConfig, cuda: bool):
    """Return a BitsAndBytesConfig for QLoRA, or None when 4-bit is unavailable."""
    if not (config.quant.load_in_4bit and cuda):
        if config.quant.load_in_4bit and not cuda:
            logger.warning("4-bit requested but no CUDA -> loading in fp32 (smoke only).")
        return None
    # Imported lazily: bitsandbytes is a CUDA-only [gpu] extra.
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_DTYPES.get(config.quant.bnb_4bit_compute_dtype, torch.bfloat16),
    )


def train(config: PipelineConfig, smoke: bool = False) -> str:
    """Run training and return the directory the adapter was saved to."""
    set_seed(config.seed)
    cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s | base model: %s", cuda, config.base_model)

    quant_config = _build_quant_config(config, cuda)

    # --- Tokenizer ---------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model, revision=config.model_revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Base model --------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        revision=config.model_revision,
        quantization_config=quant_config,
        device_map={"": 0} if cuda else None,
        torch_dtype=torch.bfloat16 if cuda else torch.float32,
    )
    model.config.use_cache = False  # required with gradient checkpointing

    if quant_config is not None:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.training.gradient_checkpointing
        )

    peft_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,  # type: ignore[arg-type]
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )

    # --- Data --------------------------------------------------------------
    proc = config.dataset.processed_dir
    data_files = {"train": f"{proc}/train.jsonl", "val": f"{proc}/val.jsonl"}
    dsets = load_dataset("json", data_files=data_files)

    # --- Trainer args ------------------------------------------------------
    tcfg = config.training
    # Force CPU-safe choices when there is no GPU.
    optim = tcfg.optim if cuda else "adamw_torch"
    bf16 = tcfg.bf16 and cuda
    fp16 = tcfg.fp16 and cuda
    grad_ckpt = tcfg.gradient_checkpointing and cuda

    # SFTConfig / SFTTrainer keyword names have drifted across trl versions
    # (e.g. max_seq_length -> max_length, tokenizer -> processing_class). Build the
    # kwargs, then filter/rename to whatever the installed trl actually accepts so
    # the same code runs on the pinned (3.11) lock and on newer trl.
    import inspect

    sft_kwargs = dict(
        output_dir=str(config.trainer_dir),
        dataset_text_field="text",
        max_seq_length=tcfg.max_seq_length,
        packing=tcfg.packing,
        num_train_epochs=tcfg.num_train_epochs,
        max_steps=3 if smoke else -1,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        learning_rate=tcfg.learning_rate,
        warmup_ratio=tcfg.warmup_ratio,
        weight_decay=tcfg.weight_decay,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        logging_steps=tcfg.logging_steps,
        save_strategy="no" if smoke else "steps",
        save_steps=tcfg.save_steps,
        optim=optim,
        gradient_checkpointing=grad_ckpt,
        bf16=bf16,
        fp16=fp16,
        seed=config.seed,
        report_to=[] if config.report_to == "none" else [config.report_to],
        run_name=config.run_name,
    )
    sft_params = set(inspect.signature(SFTConfig.__init__).parameters)
    if "max_seq_length" not in sft_params and "max_length" in sft_params:
        sft_kwargs["max_length"] = sft_kwargs.pop("max_seq_length")
    sft_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sft_params})

    trainer_kwargs = dict(
        model=model,
        args=sft_args,
        train_dataset=dsets["train"],
        eval_dataset=dsets["val"],
        peft_config=peft_config,
    )
    trainer_params = set(inspect.signature(SFTTrainer.__init__).parameters)
    tok_key = "processing_class" if "processing_class" in trainer_params else "tokenizer"
    trainer_kwargs[tok_key] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)

    logger.info("Starting training (%s steps target)...", "3 (smoke)" if smoke else "full")
    trainer.train()

    adapter_dir = str(config.adapter_dir)
    trainer.save_model(adapter_dir)  # PEFT: writes adapter_config.json + adapter weights
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved adapter -> %s", adapter_dir)
    return adapter_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning")
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    parser.add_argument(
        "--smoke", action="store_true", help="Tiny CPU/MPS run (few steps) to prove the pipeline"
    )
    args = parser.parse_args()
    train(load_config(args.config), smoke=args.smoke)


if __name__ == "__main__":
    main()
