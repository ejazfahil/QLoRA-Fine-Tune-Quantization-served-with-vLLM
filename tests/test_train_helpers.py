import torch

from common.config import load_config
from train.train import _build_quant_config


def test_quant_config_disabled_without_cuda():
    """On a CPU-only box, QLoRA 4-bit must degrade to None (no bitsandbytes)."""
    cfg = load_config("configs/mistral7b_qlora.yaml")
    assert _build_quant_config(cfg, cuda=False) is None


def test_quant_config_none_when_not_requested():
    cfg = load_config("configs/smoke.yaml")  # load_in_4bit: false
    assert _build_quant_config(cfg, cuda=torch.cuda.is_available()) is None
