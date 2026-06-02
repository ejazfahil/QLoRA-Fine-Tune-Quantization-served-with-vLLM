"""Merge a trained LoRA adapter into the base model and save standalone weights.

Merge-vs-serve-adapter tradeoff (documented in the README):
  - **Serve adapter** (vLLM ``--enable-lora``): one base in VRAM, hot-swap many
    adapters, tiny artifacts. Slight per-token overhead; base+adapter must match.
  - **Merge** (this script): a single self-contained checkpoint, zero adapter
    overhead, and a prerequisite for AWQ/GPTQ quantization. Costs full-model
    disk per variant and loses multi-adapter flexibility.

Load the base in fp16/bf16 (NOT 4-bit) so the merge is lossless, then
``merge_and_unload``.

    python -m export.merge_weights --config configs/mistral7b_qlora.yaml
"""

from __future__ import annotations

import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.config import PipelineConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("merge")


def merge(config: PipelineConfig) -> str:
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading base %s in %s", config.base_model, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=dtype, device_map="auto" if torch.cuda.is_available() else None
    )

    logger.info("Attaching adapter %s", config.adapter_dir)
    model = PeftModel.from_pretrained(base, str(config.adapter_dir))

    logger.info("Merging adapter into base weights (merge_and_unload)...")
    model = model.merge_and_unload()

    out = str(config.merged_dir)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(config.base_model).save_pretrained(out)
    logger.info("Saved merged model -> %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    merge(load_config(args.config))


if __name__ == "__main__":
    main()
