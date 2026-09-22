"""Fixed-seed decode benchmark for MTP tuning against an idle local service.

Configuration changes belong to the backend startup, not request sampling.
Replay the same run-id after changing one backend parameter and restarting.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request


def workloads(run_id, maximum):
    rng = random.Random(98521)
    words = "river cloud stone garden pencil ocean forest bridge silver green morning mountain valley wooden village meadow".split()
    background = " ".join(rng.choices(words, k=8192))
    prompts = {
        "code": (
            "Write a complete Python implementation of a thread-safe bounded LRU cache with get and put, "
            "then write unit tests for eviction, updating an existing key, and concurrent access. "
            "Explain its time complexity and locking decisions.", "low",
        ),
        "chinese": (
            "请用中文写一篇深入浅出的文章，解释数据库事务的隔离级别。依次讨论脏读、不可重复读、"
            "幻读和写偏差，每一项给出两个并发事务的具体操作时序，再说明快照隔离与可串行化的区别。",
            "low",
        ),
        "reasoning": (
            "Find all triples of positive integers a <= b <= c such that 1/a + 1/b + 1/c = 1. "
            "Give a rigorous proof that the list is complete, explaining the bounds and case analysis.",
            "max",
        ),
        "long_context": (
            "The following is irrelevant background data:\n" + background +
            "\nEnd of background. Write a detailed explanation of how a hash table handles collisions, "
            "including chaining, open addressing, resizing, and adversarial keys. Give Python examples.",
            "low",
        ),
    }
    for temperature in (0, 1):
        for family, (prompt, effort) in prompts.items():
            name = f"{family}_t{temperature}"
            yield name, {
                "messages": [{"role": "user", "content": f"Benchmark {run_id}/{name}.\n" + prompt}],
                "temperature": temperature, "top_p": 0.95, "seed": 874,
                "reasoning_effort": effort, "max_tokens": maximum,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1235")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    key = os.environ["API_KEY"]

    def request(path, payload=None):
        req = urllib.request.Request(
            args.base_url.rstrip("/") + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as response:
            return json.load(response)

    def idle():
        data = request("/metrics")
        if data["summary"]["in_flight"] or data["server"]["request_queue_depth"]:
            raise RuntimeError("Service is busy; benchmark aborted")

    previous = json.loads(args.compare.read_text()) if args.compare else None
    if previous and previous["run_id"] != args.run_id:
        parser.error("Comparison run-id mismatch")
    result = {
        "run_id": args.run_id, "label": args.label,
        "health": request("/health"), "settings": request("/v1/settings")["current"],
        "requests": [],
    }
    idle()
    request("/v1/chat/completions", {
        "model": args.model, "messages": [{"role": "user", "content": "Reply exactly READY."}],
        "temperature": 0, "seed": 874, "reasoning_effort": "low", "max_tokens": 16,
    })
    failures = []
    for name, payload in workloads(args.run_id, args.max_tokens):
        if args.only and name not in args.only:
            continue
        idle()
        payload["model"] = args.model
        print(name + ": submitting", flush=True)
        start = time.perf_counter()
        response = request("/v1/chat/completions", payload)
        row = {
            "name": name, "seconds": time.perf_counter() - start,
            "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "usage": response["usage"], "timings": response["timings"],
            "choice": response["choices"][0],
        }
        if previous:
            old = next(r for r in previous["requests"] if r["name"] == name)
            row["same_input"] = old["input_sha256"] == row["input_sha256"]
            row["same_output"] = old["choice"] == row["choice"]
            row["decode_speedup"] = row["timings"]["predicted_per_second"] / old["timings"]["predicted_per_second"]
            if not row["same_input"] or not row["same_output"]:
                failures.append(name)
        result["requests"].append(row)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("choice", "input_sha256")}), flush=True)
    if not result["requests"]:
        parser.error("No cases selected")
    if failures:
        raise SystemExit("Output parity failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
