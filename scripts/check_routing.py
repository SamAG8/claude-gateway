#!/usr/bin/env python3
"""Is the cheap tier actually answering? Ask the deployed gateway and see.

The question this answers is not "is the gateway up" — /health says that. It is
"did a light turn get the light engine, and did a turn that must not, not". You
cannot tell from an answer's text, and a reroute is silent by design, so read the
one field that gives it away:

    the `model` the response echoes back.

The HTTP engine reports the alias the client asked for (`glm-flash`). The CLI
engine reports the real Claude id it ran (`claude-…`). So a probe that asks for
glm-flash and is answered by `claude-haiku-…` was rerouted — working as designed,
but NOT using the tier you think it is.

Usage, from anywhere:

    GATEWAY_KEY=<a PAT or the shared API_KEY> python3 scripts/check_routing.py

or on the production host, without the key ever leaving it:

    docker exec claude-gateway sh -c \
      'GATEWAY_KEY=$(grep "^API_KEY=" /run/secrets/app.env | cut -d= -f2-) \
       python3 /srv/gateway/scripts/check_routing.py'
"""
import json
import os
import sys
import time
import urllib.request

BASE = os.environ.get("GATEWAY_URL", "https://ap.constralabs.ai/llm-gateway").rstrip("/")
KEY = os.environ.get("GATEWAY_KEY", "")

# Each probe states what SHOULD happen, so the script can say pass or fail rather
# than leaving a person to interpret a model id.
PROBES = [
    ("light work", "glm-flash", None,
     "Reply with one word: the capital of Japan.", "openrouter"),
    ("heavy work", "opus", None,
     "Reply with one word: the capital of Japan.", "cli"),
    ("light work, company data attached", "glm-flash", "cap_probe_not_a_real_token",
     "Reply with one word: the capital of Japan.", "cli"),
]


def ask(model, mcp_token, text):
    body = json.dumps({
        "model": model, "max_tokens": 64, "stream": False,
        "messages": [{"role": "user", "content": text}],
    }).encode()
    headers = {"content-type": "application/json", "x-api-key": KEY,
               "anthropic-version": "2023-06-01"}
    if mcp_token:
        headers["x-mcp-token"] = mcp_token
    req = urllib.request.Request(f"{BASE}/v1/messages", data=body, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.load(resp)
    return payload, round((time.monotonic() - started) * 1000)


def engine_of(answered_model: str) -> str:
    """Which engine answered, read off the model it reported."""
    return "cli" if str(answered_model).startswith("claude-") else "openrouter"


def main() -> int:
    if not KEY:
        print("Set GATEWAY_KEY (a ConstraAP PAT, or the gateway's shared API_KEY).")
        return 2
    print(f"{BASE}\n")
    print(f"{'probe':<36}{'asked':<12}{'answered by':<12}{'ms':>7}  verdict")
    failures = 0
    for name, model, mcp, text, expected in PROBES:
        try:
            payload, ms = ask(model, mcp, text)
        except Exception as e:  # noqa: BLE001 — a probe that cannot run is a failure
            print(f"{name:<36}{model:<12}{'—':<12}{'—':>7}  ERROR {e}")
            failures += 1
            continue
        got = engine_of(payload.get("model", ""))
        ok = got == expected
        failures += not ok
        note = "ok" if ok else f"EXPECTED {expected}"
        print(f"{name:<36}{model:<12}{got:<12}{ms:>7}  {note}")

    print()
    if failures:
        print("Something is not routing as intended. Look at `route_reason` in the "
              "usage log — it names the rule that overrode the choice:\n"
              "  python3 scripts/usage_report.py $USAGE_LOG --today")
    else:
        print("Light work is answered by the cheap engine, heavy work by Claude, and "
              "company data never leaves Claude.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
