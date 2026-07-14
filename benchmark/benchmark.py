"""Async-Benchmark für einen OpenAI-kompatiblen LLM-Server (Ollama oder llama-server).

Misst je Concurrency-Stufe: TTFT, TPS, E2E-Latenz (P50/P95/P99) und aggregierten
Durchsatz. Ergebnisse als CSV in results/.

Beispiel:
  python benchmark/benchmark.py --base-url http://localhost:11434/v1 \
      --model llama3.2:3b --concurrency 1 2 4 8 --requests 16
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

PROMPT = (
    "Erkläre den Trade-off zwischen Modell-Quantisierung und Ausgabequalität "
    "bei LLM-Inferenz. Gib eine konkrete, technische Antwort."
)


@dataclass
class RequestResult:
    concurrency: int
    ok: bool
    ttft_s: float
    e2e_s: float
    completion_tokens: int
    tps: float


async def one_request(
    client: httpx.AsyncClient, base_url: str, model: str, concurrency: int, max_tokens: int
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    t_first: float | None = None
    completion_tokens = 0

    try:
        async with client.stream(
            "POST", f"{base_url}/chat/completions", json=payload, timeout=180
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                # Erstes Token-Delta = TTFT.
                for choice in chunk.get("choices", []):
                    if choice.get("delta", {}).get("content") and t_first is None:
                        t_first = time.perf_counter()
                # Genaue Token-Zahl aus dem finalen usage-Chunk.
                if chunk.get("usage", {}).get("completion_tokens"):
                    completion_tokens = chunk["usage"]["completion_tokens"]
    except (httpx.HTTPError, asyncio.TimeoutError):
        return RequestResult(concurrency, False, 0.0, 0.0, 0, 0.0)

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else (t_end - t_start)
    e2e = t_end - t_start
    decode_time = max(e2e - ttft, 1e-6)
    tps = completion_tokens / decode_time if completion_tokens else 0.0
    return RequestResult(concurrency, True, ttft, e2e, completion_tokens, tps)


async def run_level(
    base_url: str, model: str, concurrency: int, n_requests: int, max_tokens: int
) -> tuple[list[RequestResult], float]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    results: list[RequestResult] = []
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits) as client:
        async def worker() -> None:
            async with sem:
                results.append(
                    await one_request(client, base_url, model, concurrency, max_tokens)
                )

        wall_start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(n_requests)))
        wall = time.perf_counter() - wall_start

    return results, wall


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def summarize(results: list[RequestResult], wall: float) -> dict:
    ok = [r for r in results if r.ok]
    ttfts = [r.ttft_s for r in ok]
    e2es = [r.e2e_s for r in ok]
    tps = [r.tps for r in ok]
    total_tokens = sum(r.completion_tokens for r in ok)
    return {
        "concurrency": results[0].concurrency if results else 0,
        "requests": len(results),
        "ok": len(ok),
        "failed": len(results) - len(ok),
        "ttft_p50": round(pct(ttfts, 0.50), 3),
        "ttft_p95": round(pct(ttfts, 0.95), 3),
        "e2e_p50": round(pct(e2es, 0.50), 3),
        "e2e_p95": round(pct(e2es, 0.95), 3),
        "tps_per_req_mean": round(statistics.mean(tps), 1) if tps else 0.0,
        "throughput_tok_s": round(total_tokens / wall, 1) if wall else 0.0,
        "wall_s": round(wall, 2),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--requests", type=int, default=16, help="Anfragen pro Concurrency-Stufe")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--tag", default="run", help="Label für den Dateinamen (z. B. Modell/Quant)")
    args = ap.parse_args()

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"benchmark_{args.tag}_{stamp}.csv"

    summaries: list[dict] = []
    print(
        f"Benchmark {args.base_url}  model={args.model}  "
        f"{args.requests} req/Stufe  max_tokens={args.max_tokens}\n"
    )
    header = f"{'conc':>5} {'ok':>4} {'ttft_p50':>9} {'ttft_p95':>9} {'e2e_p50':>8} {'tok/s':>8}"
    print(header)
    print("-" * len(header))

    for conc in args.concurrency:
        results, wall = await run_level(
            args.base_url, args.model, conc, args.requests, args.max_tokens
        )
        s = summarize(results, wall)
        summaries.append(s)
        print(
            f"{s['concurrency']:>5} {s['ok']:>4} {s['ttft_p50']:>9} "
            f"{s['ttft_p95']:>9} {s['e2e_p50']:>8} {s['throughput_tok_s']:>8}"
        )

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nGeschrieben: {out}")


if __name__ == "__main__":
    asyncio.run(main())
