"""Cold + exact-restore needle tests against the local service; saves evidence.

Requires APC_EXACT_MAX_TOKENS=0 (or >= target) for the full warm-path test.
No external requests or weight loading in this client.
"""
import argparse
import json
import os
from pathlib import Path
import random
import re
import time
import urllib.request
import urllib.error
import uuid

from transformers import AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=200000)
parser.add_argument("--model-path", type=Path,
                    default=Path(os.environ.get("MODEL", str(Path.home() / "models/GLM-5.3-Flash-FineTunning-MXFP8"))),
                    help="Local tokenizer directory matching the model being tested")
parser.add_argument("--output", default="diagnostics/service-long-context.json")
parser.add_argument("--resume", help="Reuse a saved run's prefix and append records")
parser.add_argument("--reuse-prefix", help="Replay a saved run's input in one new request")
parser.add_argument("--extend-to", type=int, default=256000)
parser.add_argument("--endpoint", choices=("chat/completions", "responses"), default="chat/completions")
parser.add_argument("--max-output-tokens", type=int, default=1024)
parser.add_argument("--check-budget", action="store_true")
parser.add_argument("--stream", action="store_true", help="Verify Responses SSE deltas and terminal response")
parser.add_argument("--disconnect-after", type=float, help="Close a replay stream after response.created plus this many seconds")
args = parser.parse_args()
if args.stream and args.endpoint != "responses":
    parser.error("--stream currently requires --endpoint responses")
if args.resume and args.reuse_prefix:
    parser.error("--resume and --reuse-prefix are mutually exclusive")
if args.disconnect_after is not None and (not args.stream or not args.reuse_prefix or args.disconnect_after <= 0):
    parser.error("--disconnect-after requires --stream, --reuse-prefix and a positive delay")
tokenizer = AutoTokenizer.from_pretrained(args.model_path.expanduser(), local_files_only=True)
previous = json.loads(Path(args.resume).read_text()) if args.resume else None
prefix_source = json.loads(Path(args.reuse_prefix).read_text()) if args.reuse_prefix else previous
if prefix_source and prefix_source["target_tokens"] != args.tokens:
    parser.error("--tokens must match the saved prefix's target_tokens")
nonce = prefix_source["nonce"] if prefix_source else uuid.uuid4().hex
rng = random.Random(17427)
words = "apple river cloud stone garden window pencil summer ocean forest paper bridge yellow silver green orange purple morning evening bottle mountain valley quiet bright wooden market village meadow tunnel fabric".split()
filler = " ".join(rng.choices(words, k=args.tokens))
pieces = [filler[i * len(filler) // 4:(i + 1) * len(filler) // 4] for i in range(4)]
codes = {"ALPHA": "CEDAR-7319", "BRAVO": "OTTER-2846", "CHARLIE": "MAPLE-9502"}
lead = f"Record set {nonce}. Read the following records and retrieve the three secret codes requested at the end.\n"
question = "\nEnd of records. What are the secret codes for ALPHA, BRAVO and CHARLIE? Reply only with ALPHA=code; BRAVO=code; CHARLIE=code."

def content_for(length):
    count = max(1, length // 4)
    out = lead
    for piece, (name, code) in zip(pieces, codes.items()):
        out += piece[:count] + f"\nOFFICIAL SECRET RECORD: {name} secret code is {code}. Preserve this code exactly.\n"
    return out + pieces[3][:count] + question

# Choose the text size using the actual local tokenizer, then record the
# server's authoritative prompt_tokens in each result.
lo, hi = 0, len(filler)
while hi - lo > 4:
    mid = (hi + lo) // 2
    n = len(tokenizer.encode(content_for(mid)))
    if n < args.tokens - 64:
        lo = mid
    else:
        hi = mid
content = content_for(lo)
results = previous or {"target_tokens": args.tokens, "nonce": nonce, "expected": codes, "requests": []}

def service_metrics():
    req = urllib.request.Request("http://127.0.0.1:1235/metrics", headers={
        "Authorization": "Bearer " + os.environ.get("API_KEY", "happy-coding-axm")})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)

def request(label, text, max_output_tokens=None, expected_status=200):
    payload = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}],
               "max_tokens": max_output_tokens or args.max_output_tokens, "temperature": 0, "reasoning_effort": "low", "stream": args.stream}
    if args.endpoint == "responses":
        payload.pop("messages")
        payload["input"] = text
        payload["max_output_tokens"] = payload.pop("max_tokens")
        payload["store"] = False
    req = urllib.request.Request("http://127.0.0.1:1235/v1/" + args.endpoint,
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ.get("API_KEY", "happy-coding-axm")})
    print(f"{label}: submitting {len(tokenizer.encode(text))} content tokens", flush=True)
    start = time.monotonic()
    deltas, event_types = [], []
    server_log = Path(__file__).resolve().parents[1] / "logs/glm.log"
    log_start = len(server_log.read_text(errors="replace")) if args.disconnect_after else 0
    baseline = service_metrics() if args.disconnect_after else None
    if baseline:
        assert baseline["summary"]["in_flight"] == 0 and baseline["server"]["request_queue_depth"] == 0
        baseline_active = float(re.findall(r"GPU idle: active=([0-9.]+) GiB", server_log.read_text(errors="replace"))[-1])
    disconnected_at = None
    try:
        with urllib.request.urlopen(req, timeout=3000) as response:
            status = response.status
            if args.stream:
                data = {}
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    event = json.loads(line[6:])
                    kind = event.get("type", "error" if "error" in event else "unknown")
                    event_types.append(kind)
                    if kind == "response.created" and args.disconnect_after:
                        time.sleep(args.disconnect_after)
                        disconnected_at = time.monotonic()
                        break
                    if kind == "response.output_text.delta":
                        deltas.append(event["delta"])
                    if kind == "response.completed":
                        data = event["response"]
            else:
                data = json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        data = {"error_body": error.read().decode()}
    if disconnected_at is not None:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            lines = server_log.read_text(errors="replace")[log_start:]
            metrics = service_metrics()
            if ("Generation cancelled:" in lines and "GPU idle:" in lines
                    and metrics["summary"]["in_flight"] == 0):
                evidence = [line for line in lines.splitlines() if any(marker in line for marker in (
                    "Generation queued:", "Prefill started:", "Prefill progress:", "GPU idle:", "Generation cancelled:"))]
                after_active = float(re.findall(r"GPU idle: active=([0-9.]+) GiB", lines)[-1])
                apc_delta = (metrics["server"]["apc"]["resident_bytes"] - baseline["server"]["apc"]["resident_bytes"]) / (1 << 30)
                residual = after_active - baseline_active - apc_delta
                result = {"label": "long_prefill_disconnect", "passed": residual < 0.25,
                          "seconds_after_disconnect": time.monotonic() - disconnected_at,
                          "baseline_active_gib": baseline_active, "after_active_gib": after_active,
                          "apc_delta_gib": apc_delta, "residual_growth_gib": residual,
                          "evidence": evidence, "server": metrics["server"]}
                results["requests"].append(result)
                Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
                print(json.dumps(result), flush=True)
                raise SystemExit(0 if result["passed"] else "FAIL: request memory remains above idle baseline after allowing for APC growth")
            time.sleep(0.2)
        raise SystemExit("FAIL: long disconnected request did not cancel and release its GPU state within 60s")
    answer = data.get("output_text") or ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    passed = status == expected_status and (expected_status != 200 or all(code in answer for code in codes.values()))
    if args.stream and expected_status == 200:
        passed = passed and event_types.count("response.completed") == 1 and "error" not in event_types
        passed = passed and "".join(deltas) == answer and data.get("status") == "completed"
    result = {"label": label, "endpoint": args.endpoint, "seconds": time.monotonic() - start, "status": status,
              "max_output_tokens": payload.get("max_output_tokens", payload.get("max_tokens")),
              "passed": passed, "response": data}
    if args.stream:
        result.update(stream_text="".join(deltas), event_types=event_types)
    results["requests"].append(result)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not passed:
        raise SystemExit(f"{label}: failed; see {args.output}")

if args.reuse_prefix:
    if args.disconnect_after:
        # A new suffix longer than eight tokens forces prefill after restoring
        # a long checkpoint; total input + 64 output still fits the 262144 budget.
        content += "\nAudit " + uuid.uuid4().hex[:8] + ". Check the three secret records again and repeat their exact codes."
    request("stream_replay" if args.stream else "replay", content)
elif args.resume:
    extra_count = args.extend_to - len(tokenizer.encode(content)) - 100
    if extra_count <= 0:
        raise SystemExit("--extend-to must exceed the existing prompt length")
    extra = tokenizer.decode(tokenizer.encode(filler)[:extra_count])
    request("warm_extend", content + "\nAdditional filler records follow; the original three secret records remain authoritative.\n" + extra + question)
else:
    request("cold", content)
    # More than eight uncached tokens, exercising the prefill branch after restore.
    request("warm_append", content + "\nAdditional retrieval instruction: Check all three official secret records again. Return the same three exact codes in the requested format, without any explanation or extra text.")
usage = results["requests"][-1]["response"].get("usage", {})
cached = (usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
if cached < args.tokens - 4096:
    raise SystemExit(f"Expected a long exact cache hit, got cached_tokens={cached}")
if args.check_budget:
    usage = results["requests"][0]["response"]["usage"]
    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    exact_budget = 262144 - prompt_tokens
    request("exact_context_budget", content, max_output_tokens=exact_budget)
    request("over_context_budget", content, max_output_tokens=exact_budget + 1, expected_status=400)
print("PASS: long-context retrieval with a confirmed long exact cache hit", flush=True)
