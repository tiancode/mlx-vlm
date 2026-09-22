"""Replay consecutive requests across restarts to check allocator retention.

Run against an idle service with the same run-id and APC configuration.
The short cases measure whole-request latency; long cold/warm pairs check
that limiting the allocator does not regress prefill or change cache hits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request

from benchmark_decode import workloads as decode_workloads


def workloads(run_id):
    short = [
        ("code", "Write a Python function that removes duplicates while preserving order. Explain briefly.", "low"),
        ("chinese", "用中文简要说明数据库索引如何加快查询，以及它的两项成本。", "low"),
        ("reasoning", "Find all positive integers n such that n squared plus n equals 42. Justify your answer.", "max"),
    ]
    for repeat in range(3):
        for temperature in (0, 1):
            for family, prompt, effort in short:
                name = f"short_{family}_t{temperature}_r{repeat}"
                yield name, {
                    "messages": [{"role": "user", "content": f"Benchmark {run_id}/{name}.\n{prompt}"}],
                    "temperature": temperature, "top_p": 0.95, "seed": 874,
                    "reasoning_effort": effort, "max_tokens": 64,
                }
    for name, payload in decode_workloads(run_id, 96):
        if name.startswith("long_context"):
            yield "cold_" + name, payload
            yield "warm_" + name, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1235")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--log", type=Path, default=Path(__file__).resolve().parents[1] / "logs/glm.log")
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
        metrics = request("/metrics")
        if metrics["summary"]["in_flight"] or metrics["server"]["request_queue_depth"]:
            raise RuntimeError("Service is busy; benchmark aborted")

    def idle_evidence(offset):
        # Read only newly appended idle summaries, never persist request logs.
        for _ in range(100):
            with args.log.open("rb") as log:
                log.seek(offset)
                lines = log.read().decode(errors="replace").splitlines()
            summaries = [line for line in lines if "GPU idle:" in line]
            if summaries:
                return summaries[-1]
            time.sleep(0.05)
        raise RuntimeError("No GPU idle evidence after request")

    previous = json.loads(args.compare.read_text()) if args.compare else None
    if previous and previous["run_id"] != args.run_id:
        parser.error("Comparison run-id mismatch")
    result = {"run_id": args.run_id, "label": args.label,
              "health": request("/health"), "settings": request("/v1/settings")["current"], "requests": []}
    idle()
    request("/v1/chat/completions", {
        "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Reply exactly READY."}],
        "temperature": 0, "seed": 874, "reasoning_effort": "low", "max_tokens": 16,
    })
    failures = []
    for name, payload in workloads(args.run_id):
        idle()
        payload = dict(payload, model="glm-5.3-flash")
        offset = args.log.stat().st_size
        start = time.perf_counter()
        response = request("/v1/chat/completions", payload)
        row = {
            "name": name, "seconds": time.perf_counter() - start,
            "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "usage": response["usage"], "timings": response["timings"], "choice": response["choices"][0],
            "idle": idle_evidence(offset),
        }
        cached = row["timings"]["cache_n"]
        row["cache_expected"] = cached > 0 if name.startswith("warm_") else cached == 0
        if not row["cache_expected"]:
            failures.append(name + ": cache")
        memory = re.search(r"allocator_cache_bytes=(\d+) allocator_limit_bytes=(\d+)", row["idle"])
        if memory:
            row["pool_within_limit"] = int(memory[1]) <= int(memory[2])
            if not row["pool_within_limit"]:
                failures.append(name + ": allocator limit")
        if previous:
            old = next(r for r in previous["requests"] if r["name"] == name)
            row["same_input"] = row["input_sha256"] == old["input_sha256"]
            row["same_output"] = row["choice"] == old["choice"]
            row["same_cached_tokens"] = cached == old["timings"]["cache_n"]
            if not all(row[k] for k in ("same_input", "same_output", "same_cached_tokens")):
                failures.append(name + ": parity")
        result["requests"].append(row)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("choice", "input_sha256", "usage", "timings")}), flush=True)
    if failures:
        raise SystemExit("Checks failed: " + ", ".join(failures))


if __name__ == "__main__":
    main()
