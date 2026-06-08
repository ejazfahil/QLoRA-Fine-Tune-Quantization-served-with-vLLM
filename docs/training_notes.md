# Training Notes — 2026-06-08

## QLoRA Config
- Base: Llama-3-8B-Instruct
- LoRA r=16, alpha=32, dropout=0.05
- 4-bit NF4 quantization (bnb)
- Target modules: q_proj, v_proj, k_proj, o_proj

## Training
- 2 epochs, batch size 4, grad accum 4
- LR 2e-4, cosine schedule
- Training time: ~3h on single A100 40GB

# ts:2026-06-08T15:15:00
