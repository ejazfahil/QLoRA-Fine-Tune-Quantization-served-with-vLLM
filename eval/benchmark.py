"""Base-vs-fine-tuned benchmark harness.

Two backends:

  --backend hf     Local Transformers generation. Scores the held-out test set
                   for the base model and the fine-tuned (base + LoRA adapter)
                   model, reporting ROUGE-L / exact-match plus per-request
                   latency. Runs anywhere (CPU/MPS/GPU); used for the *quality*
                   half of the table.

  --backend vllm   Hits a running vLLM OpenAI server (see serve/) for both the
                   base route and the adapter route, reporting throughput,
                   TTFT and p50/p95 latency for the *serving* half of the table.

Results are written to ``outputs/<run>/benchmark.json`` and printed as a table.

    python -m eval.benchmark --config configs/mistral7b_qlora.yaml --backend hf
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from common.config import PipelineConfig, load_config
from common.prompt import build_prompt
from eval.metrics import percentile, quality_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("benchmark")


def _load_test_rows(config: PipelineConfig) -> list[dict]:
    path = Path(config.dataset.processed_dir) / "test.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    n = config.eval.num_samples
    return rows[:n] if n else rows


# --------------------------------------------------------------------------- #
# HF backend (quality)
# --------------------------------------------------------------------------- #
def _hf_generate(model, tokenizer, prompt: str, max_new_tokens: int) -> tuple[str, float]:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    latency = time.perf_counter() - t0
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip(), latency


def _score_hf_model(config, rows, adapter_dir: str | None, tag: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[%s] loading model (adapter=%s)", tag, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model, torch_dtype=torch.float32 if not torch.cuda.is_available() else "auto"
    )
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    preds, refs, latencies = [], [], []
    for row in rows:
        prompt = build_prompt(row["instruction"], row.get("input"))
        text, lat = _hf_generate(model, tokenizer, prompt, config.eval.max_new_tokens)
        preds.append(text)
        refs.append(row["output"])
        latencies.append(lat)

    report = quality_report(preds, refs)
    report.update(
        {
            "latency_p50_s": round(percentile(latencies, 0.5), 4),
            "latency_p95_s": round(percentile(latencies, 0.95), 4),
        }
    )
    return report


def run_hf(config: PipelineConfig) -> dict:
    rows = _load_test_rows(config)
    logger.info("Scoring %d held-out examples with the HF backend", len(rows))
    return {
        "base": _score_hf_model(config, rows, adapter_dir=None, tag="base"),
        "finetuned": _score_hf_model(config, rows, adapter_dir=str(config.adapter_dir), tag="ft"),
    }


# --------------------------------------------------------------------------- #
# vLLM backend (serving throughput / latency)
# --------------------------------------------------------------------------- #
def run_vllm(config: PipelineConfig) -> dict:
    # Imported here so the HF path has no httpx/serving requirement.
    from serve.client import benchmark_endpoint

    rows = _load_test_rows(config)
    prompts = [build_prompt(r["instruction"], r.get("input")) for r in rows]
    return {
        "base": benchmark_endpoint(config, prompts, model_route=config.base_model),
        "finetuned": benchmark_endpoint(config, prompts, model_route=config.serve.adapter_name),
    }


# --------------------------------------------------------------------------- #
def _print_table(results: dict) -> None:
    print("\n=== Benchmark: base vs fine-tuned ===")
    cols = sorted({k for v in results.values() for k in v})
    header = "metric".ljust(18) + "".join(c.ljust(14) for c in results)
    print(header)
    print("-" * len(header))
    for metric in cols:
        line = metric.ljust(18)
        for variant in results:
            line += str(results[variant].get(metric, "-")).ljust(14)
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-vs-fine-tuned benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    args = parser.parse_args()
    config = load_config(args.config)

    results = run_hf(config) if args.backend == "hf" else run_vllm(config)

    out_path = Path(config.output_dir) / config.run_name / f"benchmark_{args.backend}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    _print_table(results)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
