"""Check two simultaneous MTP requests at both sampling temperatures.

Compare across backend configurations using the same run-id. A shared prefill
duration confirms that the requests actually entered the same model batch.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from threading import Barrier
import time
import urllib.request

from benchmark_decode import workloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    headers = {"Authorization": "Bearer " + os.environ["API_KEY"], "Content-Type": "application/json"}

    def request(path, payload=None):
        req = urllib.request.Request(
            "http://127.0.0.1:1235" + path,
            data=None if payload is None else json.dumps(payload).encode(), headers=headers,
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.load(response)

    old = json.loads(args.compare.read_text()) if args.compare else None
    result = {"run_id": args.run_id, "label": args.label, "batches": []}
    if old and old["run_id"] != args.run_id:
        parser.error("Comparison run-id mismatch")
    cases = dict(workloads(args.run_id, 128))
    for temperature in (0, 1):
        metrics = request("/metrics")
        assert metrics["summary"]["in_flight"] == metrics["server"]["request_queue_depth"] == 0
        names = [f"code_t{temperature}", f"chinese_t{temperature}"]
        barrier = Barrier(2)

        def generate(name):
            payload = {"model": "glm-5.3-flash", **cases[name]}
            barrier.wait(timeout=10)
            response = request("/v1/chat/completions", payload)
            return {
                "name": name,
                "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
                "usage": response["usage"], "timings": response["timings"],
                "choice": response["choices"][0],
            }

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(generate, names))
        batch = {
            "temperature": temperature, "seconds": time.perf_counter() - start,
            "shared_prefill": abs(rows[0]["timings"]["prompt_ms"] - rows[1]["timings"]["prompt_ms"]) < 1e-6,
            "requests": rows,
        }
        if old:
            previous = next(b for b in old["batches"] if b["temperature"] == temperature)
            batch["same_input"] = all(a["input_sha256"] == b["input_sha256"] for a, b in zip(previous["requests"], rows))
            batch["same_output"] = all(a["choice"] == b["choice"] for a, b in zip(previous["requests"], rows))
        result["batches"].append(batch)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in batch.items() if k != "requests"}), flush=True)
        if not batch["shared_prefill"]:
            raise SystemExit("Requests did not share a prefill; this run cannot establish batch parity")
        if old and (not batch["same_input"] or not batch["same_output"]):
            raise SystemExit("Batched output parity failed")


if __name__ == "__main__":
    main()
