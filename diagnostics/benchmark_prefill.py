"""Reproducible HTTP prefill/continuation benchmark; never resets shared APC.

Run before and after restarting the backend, using the same --run-id.
Each cold case has a different prefix; warm cases immediately extend it.
The service must be idle. No weights are loaded by this client.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import urllib.request


def workloads(run_id, model_path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    words = "apple river cloud stone garden window pencil summer ocean forest paper bridge yellow silver green orange purple morning evening bottle mountain valley quiet bright wooden market village meadow tunnel fabric".split()
    expected = ["CEDAR-7319", "OTTER-2846", "MAPLE-9502"]
    for temperature in (0, 1):
        for size in (2048, 8192, 16384):
            name = f"cold_{size}_t{temperature}"
            rng = random.Random(f"{run_id}/{name}")
            filler = " ".join(rng.choices(words, k=size))
            lead = f"Benchmark {run_id}/{name}. Retrieve the three official records below.\n"
            question = "\nReply only ALPHA=code; BRAVO=code; CHARLIE=code using the three official records."

            def content(length):
                parts = [lead]
                for i, code in enumerate(expected):
                    start = i * len(filler) // 4
                    parts.append(filler[start:start + length // 4])
                    parts.append(f"\nOFFICIAL RECORD: {('ALPHA', 'BRAVO', 'CHARLIE')[i]}={code}\n")
                parts.append(filler[3 * len(filler) // 4:3 * len(filler) // 4 + length // 4])
                return "".join(parts) + question

            lo, hi = 0, len(filler)
            while hi - lo > 4:
                mid = (lo + hi) // 2
                if len(tokenizer.encode(content(mid))) < size - 64:
                    lo = mid
                else:
                    hi = mid
            prompt = content(lo)
            yield name, prompt, temperature, 96, expected
            if size == 16384:
                yield (
                    f"warm_{size}_t{temperature}",
                    prompt + "\nCheck all three official records again and return exactly the same codes, with no explanation or additional text.",
                    temperature, 96, expected,
                )
        yield (
            f"decode_t{temperature}",
            f"Benchmark {run_id}/decode/t{temperature}. Write a Python merge sort implementation and unit tests covering empty input, duplicates, negatives, and already sorted input. Explain its time and space complexity.",
            temperature, 256, [],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:1235")
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--model-path", default=str(Path.home() / "models/GLM-5.3-Flash-FineTunning-MXFP8"))
    parser.add_argument("--run-id", required=True)
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

    previous = json.loads(args.compare.read_text()) if args.compare else None
    if previous and previous["run_id"] != args.run_id:
        parser.error("--run-id must match the comparison run")
    result = {
        "run_id": args.run_id, "model": args.model, "model_path": args.model_path,
        "health": request("/health"), "settings": request("/v1/settings")["current"],
        "requests": [],
    }
    for name, prompt, temperature, maximum, expected in workloads(args.run_id, args.model_path):
        metrics = request("/metrics")
        if metrics["summary"]["in_flight"] or metrics["server"]["request_queue_depth"]:
            raise RuntimeError("Service is busy; benchmark aborted to avoid mixed-load measurements")
        payload = {
            "model": args.model, "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": 0.95, "seed": 874,
            "reasoning_effort": "low", "max_tokens": maximum,
        }
        print(f"{name}: submitting", flush=True)
        start = time.perf_counter()
        response = request("/v1/chat/completions", payload)
        elapsed = time.perf_counter() - start
        choice = response["choices"][0]
        answer = choice["message"].get("content") or ""
        cached = response["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
        row = {
            "name": name, "input_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
            "seconds": elapsed, "usage": response["usage"], "timings": response.get("timings"),
            "choice": choice, "passed": all(code in answer for code in expected),
            "cache_expected": cached > 0 if name.startswith("warm_") else cached == 0,
        }
        if previous:
            old = next(r for r in previous["requests"] if r["name"] == name)
            row["same_input"] = old["input_sha256"] == row["input_sha256"]
            row["same_output"] = old["choice"] == choice
            row["wall_speedup"] = old["seconds"] / elapsed
        result["requests"].append(row)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k not in ("choice", "input_sha256")}), flush=True)
        if not row["passed"] or not row["cache_expected"]:
            raise RuntimeError(f"{name}: retrieval/cache check failed; see {args.output}")
        if previous and (not row["same_input"] or not row["same_output"]):
            raise RuntimeError(f"{name}: input/output parity failed; see {args.output}")


if __name__ == "__main__":
    main()
