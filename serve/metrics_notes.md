# Serving metrics: vLLM `/metrics` -> Prometheus -> Grafana

vLLM's OpenAI server exposes a Prometheus endpoint at **`/metrics`** (no flag
needed). The dashboard in `observability/grafana/vllm_serving_dashboard.json`
plots the panels the benchmark report references.

## Key metrics emitted by vLLM

| Metric | Meaning |
| --- | --- |
| `vllm:e2e_request_latency_seconds` (histogram) | end-to-end request latency; p50/p95/p99 via `histogram_quantile` |
| `vllm:time_to_first_token_seconds` (histogram) | TTFT distribution |
| `vllm:generation_tokens_total` (counter) | generated tokens; `rate(...)` = output tok/s |
| `vllm:prompt_tokens_total` (counter) | prompt tokens; `rate(...)` = input tok/s |
| `vllm:num_requests_running` / `:num_requests_waiting` (gauges) | continuous-batching queue depth |
| `vllm:gpu_cache_usage_perc` (gauge) | KV-cache / VRAM pressure proxy |

## Scrape config (Prometheus)

```yaml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets: ["vllm:8000"]
    metrics_path: /metrics
```

## Example PromQL (used by the Grafana panels)

```promql
# p95 end-to-end latency
histogram_quantile(0.95, sum(rate(vllm:e2e_request_latency_seconds_bucket[1m])) by (le))

# output throughput (tokens/sec)
sum(rate(vllm:generation_tokens_total[1m]))

# VRAM / KV-cache utilisation
vllm:gpu_cache_usage_perc
```

> VRAM is reported two ways in the benchmark: `nvidia-smi` peak memory during a
> run (absolute MiB) and `vllm:gpu_cache_usage_perc` (relative KV-cache load).
