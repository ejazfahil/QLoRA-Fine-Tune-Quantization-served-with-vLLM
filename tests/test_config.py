from pathlib import Path

import pytest

from common.config import load_config

CONFIGS = ["configs/smoke.yaml", "configs/mistral7b_qlora.yaml", "configs/llama2_7b_qlora.yaml"]


@pytest.mark.parametrize("path", CONFIGS)
def test_configs_load_and_validate(path):
    cfg = load_config(path)
    assert cfg.base_model
    assert cfg.lora.r > 0
    assert 0 <= cfg.dataset.val_fraction < 1
    assert 0 <= cfg.dataset.test_fraction < 1
    assert cfg.dataset.val_fraction + cfg.dataset.test_fraction < 1


def test_derived_paths():
    cfg = load_config("configs/mistral7b_qlora.yaml")
    assert cfg.adapter_dir == Path("outputs/mistral7b-dolly-qlora/adapter")
    assert cfg.merged_dir.name == "merged"
    assert cfg.quantized_dir.name == "quantized-awq"


def test_qlora_defaults_are_nf4_double_quant():
    cfg = load_config("configs/mistral7b_qlora.yaml")
    assert cfg.quant.load_in_4bit is True
    assert cfg.quant.bnb_4bit_quant_type == "nf4"
    assert cfg.quant.bnb_4bit_use_double_quant is True
