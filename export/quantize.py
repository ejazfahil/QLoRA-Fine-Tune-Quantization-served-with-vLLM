"""Quantize the merged model to 4-bit AWQ for high-throughput vLLM serving.

AWQ (Activation-aware Weight Quantization) calibrates on a small sample of real
prompts, then packs weights to 4-bit. vLLM loads the result directly with
``--quantization awq``.

Requires CUDA + ``uv sync --extra gpu`` (AutoAWQ). Run *after* merge_weights.

    python -m export.quantize --config configs/mistral7b_qlora.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from common.config import PipelineConfig, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("quantize")


def _calibration_texts(config: PipelineConfig) -> list[str]:
    """Draw calibration prompts from the train split."""
    path = Path(config.dataset.processed_dir) / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r["text"] for r in rows[: config.awq.calib_samples]]


def quantize(config: PipelineConfig) -> str:
    # Imported lazily: AutoAWQ is a CUDA-only [gpu] extra.
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    merged = str(config.merged_dir)
    logger.info("Loading merged model %s for AWQ", merged)
    model = AutoAWQForCausalLM.from_pretrained(merged)
    tokenizer = AutoTokenizer.from_pretrained(merged, trust_remote_code=True)

    quant_cfg = {
        "zero_point": config.awq.zero_point,
        "q_group_size": config.awq.group_size,
        "w_bit": config.awq.bits,
        "version": "GEMM",
    }
    logger.info("Quantizing (%s-bit, group_size=%s)...", config.awq.bits, config.awq.group_size)
    model.quantize(tokenizer, quant_config=quant_cfg, calib_data=_calibration_texts(config))

    out = str(config.quantized_dir)
    model.save_quantized(out)
    tokenizer.save_pretrained(out)
    logger.info("Saved AWQ model -> %s (serve with --quantization awq)", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="AWQ 4-bit quantization of the merged model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    quantize(load_config(args.config))


if __name__ == "__main__":
    main()
