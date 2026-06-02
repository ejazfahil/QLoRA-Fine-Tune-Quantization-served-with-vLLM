"""Quality + serving metrics used by the benchmark harness.

Quality (held-out task set):
  - ROUGE-L F1   : lexical overlap with the reference answer.
  - exact_match  : normalised string equality (strict).

Serving (under vLLM):
  - throughput (tok/s), TTFT, and p50/p95 latency are computed in
    ``serve/client.py`` and merged into the same results table.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Sequence


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    hits = sum(_normalise(p) == _normalise(r) for p, r in zip(predictions, references, strict=False))
    return hits / len(predictions)


def rouge_l(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean ROUGE-L F1. Uses the ``rouge_score`` package (no network)."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = [
        scorer.score(r, p)["rougeL"].fmeasure
        for p, r in zip(predictions, references, strict=False)
    ]
    return statistics.mean(scores) if scores else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def quality_report(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    return {
        "rougeL_f1": round(rouge_l(predictions, references), 4),
        "exact_match": round(exact_match(predictions, references), 4),
        "n": float(len(predictions)),
    }
