"""Live, serial checks through the public proxy for harness-style requests."""
import json
import os
from pathlib import Path
import time
import uuid

import httpx

root = Path(__file__).resolve().parents[1]
log = root / "logs/glm.log"
results = []
headers = {"Authorization": "Bearer " + os.environ.get("API_KEY", "happy-coding-axm")}


def payload(text, stream=False):
    return {"model": "glm-5.3-flash", "input": text, "stream": stream,
            "max_output_tokens": 4096, "reasoning_effort": "low", "temperature": 0}


def wait_idle(client, start, cancelled=False):
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        lines = log.read_text(errors="replace")[start:]
        dropped = "Dropped cancelled queued request=" in lines
        cancellation = "Generation cancelled:" in lines or dropped
        idle = "GPU idle:" in lines or dropped
        metrics = client.get("/metrics").json()
        if idle and (not cancelled or cancellation) and metrics["summary"]["in_flight"] == 0:
            return [line for line in lines.splitlines() if any(marker in line for marker in ("GPU idle:", "Generation cancelled:", "Dropped cancelled queued"))]
        time.sleep(0.2)
    raise AssertionError("Request did not reach GPU idle/cancelled state within 45s")


def record(name, since, lines):
    entry = {"case": name, "seconds": round(time.monotonic() - since, 3), "evidence": lines}
    results.append(entry)
    (root / "diagnostics/lifecycle-service.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(entry), flush=True)


with httpx.Client(base_url="http://127.0.0.1:1235", headers=headers, timeout=120) as client:
    for _ in range(180):
        try:
            if client.get("/health", timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    else:
        raise SystemExit("Service not ready")

    start = len(log.read_text(errors="replace"))
    since = time.monotonic()
    body = payload("Reply with exactly PONG.")
    body["max_output_tokens"] = 512
    body["store"] = False
    response = client.post("/v1/responses", json=body)
    response.raise_for_status()
    assert "PONG" in response.json()["output_text"], response.text
    record("normal_completion", since, wait_idle(client, start))

    start = len(log.read_text(errors="replace"))
    text = uuid.uuid4().hex + "\n" + "river forest cloud garden " * 3000 + "\nWrite a long numbered list of facts about forests."
    with client.stream("POST", "/v1/responses", json=payload(text, stream=True)) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if "response.created" in line:
                time.sleep(3)  # Let prefill begin, then emulate harness exit.
                break
    since = time.monotonic()
    record("stream_disconnect_during_prefill", since, wait_idle(client, start, cancelled=True))

    start = len(log.read_text(errors="replace"))
    text = uuid.uuid4().hex + " List the integers from 1 to 10000, one per line, without skipping any."
    try:
        client.post("/v1/responses", json=payload(text), timeout=httpx.Timeout(0.8, connect=5))
        raise AssertionError("The nonstream request unexpectedly completed before disconnect")
    except httpx.ReadTimeout:
        pass
    since = time.monotonic()
    record("nonstream_disconnect", since, wait_idle(client, start, cancelled=True))

    start = len(log.read_text(errors="replace"))
    response_id = None
    seen_delta = False
    completed = False
    with client.stream("POST", "/v1/responses", json=payload(text + " Continue until 10000.", stream=True)) as response:
        response.raise_for_status()
        lines = response.iter_lines()
        for line in lines:
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            if event.get("type") == "response.created":
                response_id = event["response"]["id"]
            if event.get("type", "").endswith(".delta"):
                seen_delta = True
                since = time.monotonic()
                cancelled = client.post(f"/v1/responses/{response_id}/cancel")
                cancelled.raise_for_status()
                assert cancelled.json()["status"] == "cancelled"
                break
        assert response_id and seen_delta
        # The response must not report successful completion after cancellation.
        for line in lines:
            if "response.completed" in line:
                completed = True
    assert not completed
    evidence = wait_idle(client, start, cancelled=True)
    assert client.get(f"/v1/responses/{response_id}").json()["status"] == "cancelled"
    assert client.delete(f"/v1/responses/{response_id}").json()["deleted"]
    record("explicit_response_cancel", since, evidence)

print("PASS: completion, stream disconnect, nonstream disconnect, explicit cancel", flush=True)
