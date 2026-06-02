.PHONY: help setup data smoke train eval merge quantize serve lint typecheck test check clean

UV ?= uv

help:
	@echo "Targets:"
	@echo "  setup      Create the locked Python 3.11 env (CPU/dev deps)"
	@echo "  data       Build the instruction dataset splits (seeded)"
	@echo "  smoke      Run the tiny-model CPU/MPS pipeline smoke test"
	@echo "  train      QLoRA train from a config (GPU):  make train CONFIG=configs/mistral7b_qlora.yaml"
	@echo "  eval       Base-vs-finetuned benchmark:      make eval CONFIG=configs/mistral7b_qlora.yaml"
	@echo "  merge      Merge adapter into base (GPU)"
	@echo "  quantize   AWQ 4-bit quantize merged model (GPU)"
	@echo "  serve      Launch vLLM OpenAI server with the adapter (GPU)"
	@echo "  lint       ruff check"
	@echo "  typecheck  mypy"
	@echo "  test       pytest"
	@echo "  check      lint + typecheck + test"

setup:
	$(UV) sync --extra dev

data:
	$(UV) run python -m data.prepare_data --config configs/smoke.yaml

smoke:
	$(UV) run python -m train.train --config configs/smoke.yaml --smoke

train:
	$(UV) run python -m train.train --config $(CONFIG)

eval:
	$(UV) run python -m eval.benchmark --config $(CONFIG)

merge:
	$(UV) run python -m export.merge_weights --config $(CONFIG)

quantize:
	$(UV) run python -m export.quantize --config $(CONFIG)

serve:
	bash serve/launch_vllm.sh $(CONFIG)

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy train eval serve export data

test:
	$(UV) run pytest

check: lint typecheck test

clean:
	rm -rf outputs adapters merged quantized data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
