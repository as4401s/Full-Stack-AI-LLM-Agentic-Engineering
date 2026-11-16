"""mock_server.py - a tiny local stand-in for a model API, so every lesson example is free.

What it does: answers POST /v1/<provider>/chat after a fixed delay, like a model that needs a few
seconds to think. Two special providers help us practise reliability:
  * "flaky"   fails with HTTP 503 a few times per request, then succeeds (practise retries).
  * "billing" counts one charge for every real execution and can lose its first reply
              (practise idempotency keys). GET /v1/billing/ledger shows the charges.
What it needs: only the Python standard library. Run it with  python -m caredesk_fanout.mock_server
What to expect: JSON replies shaped like ModelReply in models.py.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PHRASES = [
    "Prioritise the chest pain referral before the routine review.",
    "Two follow-up scans are overdue for this synthetic patient.",
    "The discharge summary is complete and ready for sign-off.",
    "Suggest a medication reconciliation at the next visit.",
]


class State:
    """Everything the server remembers. A lock protects it because requests run in threads."""

    def __init__(self, delay_s: float, flaky_failures: int, lose_reply_s: float) -> None:
        self.delay_s = delay_s
        self.flaky_failures = flaky_failures      # 503s before a flaky request succeeds
        self.lose_reply_s = lose_reply_s          # how long the billing server stalls its first reply
        self.lock = threading.Lock()
        self.flaky_seen: dict[str, int] = {}      # request identity -> failures served so far
        self.saved: dict[str, dict] = {}          # idempotency key -> saved reply
        self.charges = 0
        self.executions = 0
        self.lose_replies_left = 0                # how many billing replies to stall

    def reset_billing(self, lose: int) -> None:
        with self.lock:
            self.saved.clear()
            self.charges = 0
            self.executions = 0
            self.lose_replies_left = lose


def make_reply(provider: str, prompt: str, max_tokens: int) -> dict:
    """A reply shaped like a real model API: text plus token usage."""
    words = max(1, len(prompt.split()))
    return {
        "model": f"{provider}-mini",
        "text": f"[{provider}] {random.choice(PHRASES)}",
        "usage": {"prompt_tokens": words, "completion_tokens": min(max_tokens, 12)},
    }


def make_handler(state: State) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:       # keep the terminal quiet
            pass

        def _send(self, code: int, payload: dict, extra: dict | None = None) -> None:
            body = json.dumps(payload).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass                                  # the client gave up (a timeout): fine

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send(200, {"ok": True})
            elif path == "/v1/billing/ledger":
                with state.lock:
                    self._send(200, {"charges": state.charges, "executions": state.executions})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            if path == "/v1/billing/reset":
                state.reset_billing(int(body.get("lose", 0)))
                return self._send(200, {"ok": True})
            parts = path.strip("/").split("/")
            if len(parts) != 3 or parts[0] != "v1" or parts[2] != "chat":
                return self._send(404, {"error": "not found"})
            provider = parts[1]
            prompt, max_tokens = body.get("prompt", ""), int(body.get("max_tokens", 64))
            key = self.headers.get("Idempotency-Key")
            if provider == "flaky":
                return self._flaky(key or prompt, provider, prompt, max_tokens)
            if provider == "billing":
                return self._billing(key, provider, prompt, max_tokens)
            time.sleep(state.delay_s)                 # the model "thinks"
            self._send(200, make_reply(provider, prompt, max_tokens))

        def _flaky(self, identity: str, provider: str, prompt: str, max_tokens: int) -> None:
            with state.lock:
                seen = state.flaky_seen.get(identity, 0)
                if seen < state.flaky_failures:
                    state.flaky_seen[identity] = seen + 1
                    return self._send(503, {"error": "temporarily unavailable"})
            self._send(200, make_reply(provider, prompt, max_tokens))

        def _billing(self, key: str | None, provider: str, prompt: str, max_tokens: int) -> None:
            with state.lock:
                if key and key in state.saved:        # a repeat of a request we already handled
                    return self._send(200, state.saved[key], {"Idempotent-Replay": "true"})
                state.executions += 1                 # real work happens, and is charged
                state.charges += 1
                reply = make_reply(provider, prompt, max_tokens)
                if key:
                    state.saved[key] = reply
                stall = state.lose_replies_left > 0
                if stall:
                    state.lose_replies_left -= 1
            if stall:
                time.sleep(state.lose_reply_s)        # the work is done, but the reply arrives too late
            self._send(200, reply)

    return Handler


def build_server(port: int = 0, delay_s: float = 3.0, flaky_failures: int = 2, lose_reply_s: float = 1.5) -> ThreadingHTTPServer:
    """Create the server (port 0 picks a free port). Call serve_forever() to run it."""
    state = State(delay_s, flaky_failures, lose_reply_s)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    server.daemon_threads = True
    server.state = state  # type: ignore[attr-defined]
    return server


def start_in_thread(**kwargs) -> tuple[ThreadingHTTPServer, str]:
    """Start the server in a background thread and return (server, base_url). Handy for demos and tests."""
    server = build_server(**kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def main() -> None:
    """Command line entry point: python -m caredesk_fanout.mock_server --delay 3"""
    ap = argparse.ArgumentParser(description="Local mock model API for the CareDesk lessons")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=3.0, help="seconds each normal call takes")
    ap.add_argument("--flaky-failures", type=int, default=2)
    ap.add_argument("--lose-reply", type=float, default=1.5)
    args = ap.parse_args()
    server = build_server(args.port, args.delay, args.flaky_failures, args.lose_reply)
    print(f"mock model API on http://127.0.0.1:{args.port}  (delay {args.delay} s)  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
