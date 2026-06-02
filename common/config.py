"""Typed, validated configuration loaded from ``configs/*.yaml``.

A single config drives the whole pipeline (data -> train -> eval -> merge ->
quantize -> serve) so a run is reproducible from one file plus a seed plus the
uv lockfile. See ``configs/`` for concrete examples.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class DatasetConfig(BaseModel):
    """Where the instruction data comes from and how it is split."""

    hf_name: str = "databricks/databricks-dolly-15k"
    hf_config: str | None = None
    hf_split: str = "train"
    # Field names in the source dataset mapped onto our schema.
    instruction_field: str = "instruction"
    input_field: str = "context"
    output_field: str = "response"
    # Cap the number of examples (useful for smoke runs / quick benchmarks).
    max_samples: int | None = None
    # Held-out split fractions. train = 1 - val - test.
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    processed_dir: str = "data/processed"


class LoraConfigModel(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class QuantConfig(BaseModel):
    """bitsandbytes 4-bit (QLoRA) settings, used only when CUDA is available."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"


class TrainingConfig(BaseModel):
    learning_rate: float = 2e-4
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 1024
    packing: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 200
    optim: str = "paged_adamw_8bit"
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42


class AwqConfig(BaseModel):
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    # Number of calibration samples drawn from the train split.
    calib_samples: int = 128


class EvalConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.0
    # Number of held-out test examples to score (None = all).
    num_samples: int | None = 200


class ServeConfig(BaseModel):
    adapter_name: str = "ft"
    max_lora_rank: int = 16
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    port: int = 8000
    dtype: str = "auto"
    quantization: str | None = None  # e.g. "awq" when serving the quantized model


class PipelineConfig(BaseModel):
    """Top-level config object."""

    run_name: str = "run"
    # Base model id. Mistral-7B-v0.3 (open) by default; Llama-2 is gated.
    base_model: str = "mistralai/Mistral-7B-v0.3"
    model_revision: str | None = None
    output_dir: str = "outputs"
    seed: int = 42
    report_to: str = "none"  # "none" | "wandb" | "trackio" | "tensorboard"

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    awq: AwqConfig = Field(default_factory=AwqConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    serve: ServeConfig = Field(default_factory=ServeConfig)

    # --- Derived paths -----------------------------------------------------
    @property
    def adapter_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "adapter"

    @property
    def merged_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "merged"

    @property
    def quantized_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "quantized-awq"

    @property
    def trainer_dir(self) -> Path:
        return Path(self.output_dir) / self.run_name / "trainer"


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a YAML pipeline config."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig(**data)
