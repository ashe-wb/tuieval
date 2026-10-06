"""A tiny OpenAI-compatible chat server for tests: no model, answers from the packs' own tests.

    python tests/mock_server.py --port 18090 --packs path/to/packs [--mode oracle|wrong|fixed] [--delay 0.2]

oracle  answers each question with its test's `reference` (or the expected tool call), so a
        correct grader passes it
wrong   answers with the test's first `wrong` entry (or "ANSWER: 0")
fixed   always "ANSWER: 42"

Streams like llama.cpp: reasoning_content then content deltas, usage and timings at the end.
"""
import argparse
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml


def load_answers(packs_dir):
    """{user message: test} for every test in every pack folder."""
    out = {}
    for name in sorted(os.listdir(packs_dir)):
        folder = os.path.join(packs_dir, name)
        for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
            if f.endswith((".yaml", ".yml")) and f not in ("tools.yaml",):
                with open(os.path.join(folder, f)) as fh:
                    for t in yaml.safe_load(fh) or []:
                        if isinstance(t, dict) and "input" in t:
                            out[" ".join(str(t["input"]).split())] = t
    return out


def reply_for(test, mode):
    """(text, tool_calls) to send for a test."""
    if mode == "fixed" or test is None:
        return "ANSWER: 42", []
    if mode == "wrong":
        return (test.get("wrong") or ["ANSWER: 0"])[0], []
    if test.get("expect_tool"):
        return "", [test["expect_tool"]]
    return test.get("reference", "ANSWER: 42"), []


class Handler(BaseHTTPRequestHandler):
    answers, mode, model, delay = {}, "oracle", "mock", 0.0

    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json({"data": [{"id": self.model}]})
        if self.path == "/health":
            return self._json({"status": "ok", "model": self.model})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/tokenize":
            return self._json({"tokens": list(range(len(str(req.get("content", "")).split())))})
        if not self.path.endswith("/chat/completions"):
            return self._json({"error": "not found"}, 404)
        users = [m for m in req.get("messages", []) if m.get("role") == "user"]
        content = users[-1]["content"] if users else ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        test = self.answers.get(" ".join(str(content).split()))
        text, calls = reply_for(test, self.mode)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        send({"choices": [{"delta": {"reasoning_content": "Thinking it over. "}}]})
        time.sleep(self.delay or 0.005)
        for i in range(0, len(text), 12):
            send({"choices": [{"delta": {"content": text[i:i + 12]}}]})
        for i, c in enumerate(calls):
            send({"choices": [{"delta": {"tool_calls": [{"index": i, "function": {
                "name": c["name"], "arguments": json.dumps(c.get("arguments") or {})}}]}}]})
        words = max(1, len(text.split()) + 3)
        send({"choices": [{"delta": {}, "finish_reason": "tool_calls" if calls else "stop"}],
              "usage": {"prompt_tokens": len(str(content).split()), "completion_tokens": words},
              "timings": {"predicted_n": words, "predicted_per_second": 50.0, "prompt_n": 10,
                          "prompt_per_second": 500.0, "cache_n": 0}})
        self.wfile.write(b"data: [DONE]\n\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=18090)
    p.add_argument("--packs", required=True)
    p.add_argument("--mode", choices=("oracle", "wrong", "fixed"), default="oracle")
    p.add_argument("--model", default="mock")
    p.add_argument("--delay", type=float, default=0.0, help="seconds each answer takes")
    a, _ = p.parse_known_args(argv)   # other flags (e.g. speed knobs under test) are accepted and ignored
    Handler.answers, Handler.mode, Handler.model, Handler.delay = load_answers(a.packs), a.mode, a.model, a.delay
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"mock server on 127.0.0.1:{a.port} ({a.mode}, {len(Handler.answers)} known questions)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
