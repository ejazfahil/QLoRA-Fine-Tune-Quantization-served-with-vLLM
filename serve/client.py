"""Thin async client for the vLLM OpenAI-compatible server + a load generator.

``benchmark_endpoint`` fires N concurrent streaming completions and reports:
  - throughput_tok_s : total generated tokens / wall-clock
  - ttft_p50_s       : time-to-first-token (p50)
  - latency_p50_s / latency_p95_s : full-response latency percentiles

The same client doubles as a manual probe:
    python -m serve.client --config configs/mistral7b_qlora.yaml \
        --prompt "Explain QLoRA in one sentence." --route mistral-ft
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time

import httpx

from common.config import PipelineConfig, load_config
from common.prompt import build_prompt
from eval.metrics import percentile


def _base_url() -> str:
    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def _api_key() -> str:
    return os.environ.get("VLLM_API_KEY", "EMPTY")


async def _stream_completion(
    client: httpx.AsyncClient, model_route: str, prompt: str, max_new_tokens: int
) -> tuple[float, float, int]:
    """Return (ttft_s, total_latency_s, generated_tokens) for one streamed request."""
    payload = {
        "model": model_route,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Authorization": f"Bearer {_api_key()}"}
    t0 = time.perf_counter()
    ttft = -1.0
    tokens = 0
    async with client.stream(
        "POST", f"{_base_url()}/completions", json=payload, headers=headers
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            if ttft < 0:
                ttft = time.perf_counter() - t0
            tokens += 1
    return ttft, time.perf_counter() - t0, tokens


def benchmark_endpoint(
    config: PipelineConfig,
    prompts: list[str],
    model_route: str,
    concurrency: int = 8,
) -> dict[str, float | str]:
    """Drive concurrent load against one model route and summarise it."""

    async def _run() -> dict[str, float | str]:
        ttfts: list[float] = []
        latencies: list[float] = []
        total_tokens = 0
        sem = asyncio.Semaphore(concurrency)
        limits = httpx.Limits(max_connections=concurrency)

        async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:

            async def _one(prompt: str) -> None:
                nonlocal total_tokens
                async with sem:
                    ttft, lat, toks = await _stream_completion(
                        client, model_route, prompt, config.eval.max_new_tokens
                    )
                    ttfts.append(ttft)
                    latencies.append(lat)
                    total_tokens += toks

            wall0 = time.perf_counter()
            await asyncio.gather(*(_one(p) for p in prompts))
            wall = time.perf_counter() - wall0

        return {
            "route": model_route,
            "requests": float(len(prompts)),
            "throughput_tok_s": round(total_tokens / wall, 2) if wall else 0.0,
            "ttft_p50_s": round(percentile(ttfts, 0.5), 4),
            "latency_p50_s": round(percentile(latencies, 0.5), 4),
            "latency_p95_s": round(percentile(latencies, 0.95), 4),
        }

    return asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM client / probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", help="Single prompt to send (probe mode)")
    parser.add_argument("--route", help="Model route: base id or adapter name")
    args = parser.parse_args()
    config = load_config(args.config)
    route = args.route or config.serve.adapter_name
    prompt = build_prompt(args.prompt or "Say hello.", None)
    result = benchmark_endpoint(config, [prompt], model_route=route, concurrency=1)
    print(result)


if __name__ == "__main__":
    main()
