#!/usr/bin/env python3
"""Quick throughput benchmark against TorchServe inference endpoint."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import numpy as np

from app.ml.torchserve_client import TorchServeClient


async def _worker(
    *,
    client: TorchServeClient,
    stop_at: float,
    model_name: str,
    seed: int,
) -> tuple[int, int, list[float]]:
    rng = np.random.default_rng(seed)
    success = 0
    errors = 0
    latencies_ms: list[float] = []

    while time.time() < stop_at:
        window = rng.standard_normal((64, 469), dtype=np.float32)
        started = time.perf_counter()
        try:
            await client.predict(window, model_name=model_name)
            success += 1
        except Exception:
            errors += 1
        finally:
            latencies_ms.append((time.perf_counter() - started) * 1000)

    return success, errors, latencies_ms


async def run_benchmark(args: argparse.Namespace) -> None:
    client = TorchServeClient(
        base_url=args.url,
        timeout_seconds=float(args.timeout_ms) / 1000.0,
        max_retries=args.retries,
    )

    deadline = time.time() + args.duration
    tasks = [
        asyncio.create_task(
            _worker(
                client=client,
                stop_at=deadline,
                model_name=args.model_name,
                seed=42 + idx,
            )
        )
        for idx in range(args.concurrent)
    ]

    started = time.perf_counter()
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started
    await client.close()

    total_success = sum(item[0] for item in results)
    total_errors = sum(item[1] for item in results)
    all_latencies = [lat for _, _, lats in results for lat in lats]
    total_requests = total_success + total_errors

    def percentile(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int((p / 100.0) * (len(ordered) - 1))
        return ordered[index]

    print("TorchServe benchmark results")
    print(f"- URL: {args.url}")
    print(f"- Model: {args.model_name}")
    print(f"- Concurrent workers: {args.concurrent}")
    print(f"- Duration: {args.duration}s")
    print(f"- Requests: {total_requests} ({total_success} success / {total_errors} errors)")
    print(f"- Throughput: {total_success / max(elapsed, 1e-6):.2f} req/s")
    if all_latencies:
        print(f"- Latency mean: {statistics.mean(all_latencies):.2f} ms")
        print(f"- Latency p50: {percentile(all_latencies, 50):.2f} ms")
        print(f"- Latency p95: {percentile(all_latencies, 95):.2f} ms")
        print(f"- Latency p99: {percentile(all_latencies, 99):.2f} ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TorchServe prediction endpoint")
    parser.add_argument("--url", default="http://localhost:8080", help="TorchServe base URL")
    parser.add_argument("--model-name", default="signflow", help="Model name on TorchServe")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent workers")
    parser.add_argument("--duration", type=int, default=20, help="Benchmark duration in seconds")
    parser.add_argument("--timeout-ms", type=int, default=2000, help="HTTP timeout in milliseconds")
    parser.add_argument("--retries", type=int, default=0, help="Retry attempts per request")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmark(parse_args()))
