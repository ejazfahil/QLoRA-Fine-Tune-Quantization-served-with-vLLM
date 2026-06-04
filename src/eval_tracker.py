"""Eval metrics. 2026-06-04"""
from dataclasses import dataclass, field
from typing import List, Dict
import json

@dataclass
class EvalSnapshot:
    step: int
    loss: float
    perplexity: float
    bleu: float = 0.0
    rouge_l: float = 0.0

@dataclass
class EvalTracker:
    snapshots: List[EvalSnapshot] = field(default_factory=list)

    def record(self, step, loss, perplexity, **kw):
        self.snapshots.append(EvalSnapshot(step, loss, perplexity, **kw))

    def best_loss(self): return min(s.loss for s in self.snapshots) if self.snapshots else float("inf")
    def to_json(self): return json.dumps([{"step":s.step,"loss":s.loss,"ppl":s.perplexity}
                                          for s in self.snapshots], indent=2)
