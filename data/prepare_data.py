"""Download a standard HF instruction dataset, normalise it to our schema, and
write seeded train/val/test JSONL splits.

Schema (one JSON object per line) -- see ``data/schema.md``:
    {
      "instruction": str,   # the task / question
      "input":       str,   # optional supporting context ("" if none)
      "output":      str,   # the reference answer
      "text":        str,   # rendered Alpaca prompt + answer (training target)
    }

Usage:
    python -m data.prepare_data --config configs/mistral7b_qlora.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from common.config import PipelineConfig, load_config
from common.prompt import build_example_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_data")


def _normalise(example: dict, cfg) -> dict:
    instruction = (example.get(cfg.instruction_field) or "").strip()
    context = (example.get(cfg.input_field) or "").strip()
    output = (example.get(cfg.output_field) or "").strip()
    return {
        "instruction": instruction,
        "input": context,
        "output": output,
        "text": build_example_text(instruction, context, output),
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(config: PipelineConfig) -> dict[str, int]:
    """Build the splits and return the row count per split."""
    dcfg = config.dataset
    logger.info("Loading %s (split=%s)", dcfg.hf_name, dcfg.hf_split)
    ds = load_dataset(dcfg.hf_name, dcfg.hf_config, split=dcfg.hf_split)

    # Drop rows missing instruction or output, then normalise.
    ds = ds.map(lambda ex: _normalise(ex, dcfg), remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: bool(ex["instruction"]) and bool(ex["output"]))

    if dcfg.max_samples is not None:
        ds = ds.shuffle(seed=config.seed).select(range(min(dcfg.max_samples, len(ds))))

    # Deterministic split driven by the global seed.
    ds = ds.shuffle(seed=config.seed)
    n = len(ds)
    n_test = int(n * dcfg.test_fraction)
    n_val = int(n * dcfg.val_fraction)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"Split leaves no training rows (n={n}). Lower val/test fractions.")

    splits = {
        "train": ds.select(range(n_train)),
        "val": ds.select(range(n_train, n_train + n_val)),
        "test": ds.select(range(n_train + n_val, n)),
    }

    out_dir = Path(dcfg.processed_dir)
    counts: dict[str, int] = {}
    for name, split in splits.items():
        rows = list(split)
        _write_jsonl(rows, out_dir / f"{name}.jsonl")
        counts[name] = len(rows)
        logger.info("Wrote %s rows -> %s", len(rows), out_dir / f"{name}.jsonl")

    # Refresh the committed sample so the schema stays self-documenting.
    sample = list(splits["train"].select(range(min(5, counts["train"]))))
    _write_jsonl(sample, Path("data/sample.jsonl"))

    meta = {
        "dataset": dcfg.hf_name,
        "seed": config.seed,
        "counts": counts,
        "val_fraction": dcfg.val_fraction,
        "test_fraction": dcfg.test_fraction,
    }
    (out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("Split metadata: %s", meta)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare instruction dataset splits.")
    parser.add_argument("--config", required=True, help="Path to a configs/*.yaml file")
    args = parser.parse_args()
    prepare(load_config(args.config))


if __name__ == "__main__":
    main()
