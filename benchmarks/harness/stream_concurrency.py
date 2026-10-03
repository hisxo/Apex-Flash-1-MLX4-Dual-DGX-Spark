#!/usr/bin/env python3
"""Measure isolated SSE request groups against a TensorFold endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any

COUNTERS = (
    "requests_total",
    "prompt_tokens_total",
    "completion_tokens_total",
    "prefill_seconds_total",
    "decode_seconds_total",
    "cached_tokens_total",
    "rounds_total",
    "drafted_total",
    "accepted_total",
)


def read_json(url: str, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def stream_request(
    url: str,
    api_key: str,
    model: str,
    level: int,
    index: int,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Transport case {level}-{index}. Return exactly 220 repetitions of the word "
                    "TOKEN separated by single spaces. No preface or punctuation."
                ),
            }
        ],
        "max_tokens": 256,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "reasoning_effort": "low",
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )

    barrier.wait()
    started = time.perf_counter()
    first_sse = None
    first_output = None
    first_content = None
    usage = None
    events = 0
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            now = time.perf_counter()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            first_sse = first_sse or now
            event = json.loads(line[6:])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                content = delta.get("content")
                if reasoning or content:
                    first_output = first_output or now
                    events += 1
                if content:
                    first_content = first_content or now
    ended = time.perf_counter()
    return {
        "index": index,
        "wall_seconds": ended - started,
        "tt_first_sse_seconds": None if first_sse is None else first_sse - started,
        "tt_first_output_seconds": None if first_output is None else first_output - started,
        "tt_first_content_seconds": None if first_content is None else first_content - started,
        "sse_output_events": events,
        "usage": usage,
    }


def subtract(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int | float]:
    names = {name.removesuffix("_total"): after[name] - before[name] for name in COUNTERS}
    names["uncached_prompt_tokens"] = names["prompt_tokens"] - names["cached_tokens"]
    return names


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default="1,2,4")
    args = parser.parse_args()

    base_url = os.environ["LOCAL_OPENAI_BASE_URL"].rstrip("/")
    api_key = os.environ.get("LOCAL_OPENAI_API_KEY", "local")
    model = os.environ.get("LOCAL_OPENAI_MODEL", "apex-flash-1-abliterated")
    health_url = f"{base_url.removesuffix('/v1')}/health"
    chat_url = f"{base_url}/chat/completions"
    results = []

    for level in (int(item) for item in args.levels.split(",")):
        before = read_json(health_url, api_key)
        if before.get("busy"):
            raise RuntimeError("server must be idle before each group")

        barrier = threading.Barrier(level)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(stream_request, chat_url, api_key, model, level, index, barrier)
                for index in range(level)
            ]
            requests = [future.result() for future in futures]
        wall_seconds = time.perf_counter() - started
        after = read_json(health_url, api_key)
        delta = subtract(after, before)
        ttft = [item["tt_first_content_seconds"] for item in requests]

        results.append(
            {
                "concurrent_clients": level,
                "group_wall_seconds": wall_seconds,
                "requests": requests,
                "tensorfold_before": {name: before[name] for name in COUNTERS},
                "tensorfold_after": {name: after[name] for name in COUNTERS},
                "tensorfold_delta": delta,
                "derived": {
                    "server_decode_tokens_per_second": ratio(
                        delta["completion_tokens"], delta["decode_seconds"]
                    ),
                    "uncached_prefill_tokens_per_second": ratio(
                        delta["uncached_prompt_tokens"], delta["prefill_seconds"]
                    ),
                    "cache_read_percent": 100
                    * ratio(delta["cached_tokens"], delta["prompt_tokens"]),
                    "completion_tokens_per_group_wall_second": ratio(
                        delta["completion_tokens"], wall_seconds
                    ),
                    "dflash2_acceptance_percent": 100 * ratio(delta["accepted"], delta["drafted"]),
                    "visible_ttft_min_seconds": min(ttft),
                    "visible_ttft_median_seconds": statistics.median(ttft),
                    "visible_ttft_max_seconds": max(ttft),
                },
            }
        )

    print(json.dumps({"schema_version": 1, "groups": results}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
