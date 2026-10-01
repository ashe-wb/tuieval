"""Standalone tool: watch the reasoning of any app that talks to your model server.

The evals don't need this (the engine streams from the server itself). It's for seeing what a
model is thinking while some *other* client uses it: point that client at the proxy instead of
the server. Each request is sent upstream as a stream, reasoning and answer are printed to your
terminal as they are generated, and the assembled (non-streaming) response is handed back to
the client, so the client sees an ordinary reply.

    tuieval watch --upstream http://localhost:8080 --port 9000

Then point the client at http://localhost:9000/v1. Every request is also saved in full to
logs/live/. With several requests in flight, one is shown live and the others print a one-line
summary when they finish (their full text is still saved).
"""
import argparse
import datetime
import http.client
import itertools
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DIM, BOLD, CYAN, RED, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[31m", "\033[0m"


class Request:
    """One chat completion passing through the proxy."""

    def __init__(self, n, model, messages):
        self.n, self.model = n, model
        users = [m for m in messages if m.get("role") == "user"]
        content = users[-1]["content"] if users else ""
        if isinstance(content, list):  # text + image parts: keep the text
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        self.text = str(content)
        self.title = " ".join(self.text.split())[:90]
        self.started = time.time()
        self.reasoning, self.answer = [], []
        self.usage, self.finish, self.error = None, "stop", None

    @property
    def tokens(self):
        return (self.usage or {}).get("completion_tokens")

    @property
    def secs(self):
        return time.time() - self.started


class Listener:
    """Receives proxy events. Subclass and override what you need.

    Exceptions raised by a listener are printed and ignored, so a UI bug can't break a request.
    """

    def on_start(self, req): pass
    def on_delta(self, req, kind, text): pass  # kind is "reasoning" or "answer"
    def on_done(self, req): pass
    def on_error(self, message): print(f"watch_proxy: {message}", file=sys.stderr)


class PrintListener(Listener):
    """The original terminal view: one request streams live, the rest print a summary line."""

    def __init__(self):
        self.screen = threading.Lock()
        self.live = {}
        self.mode = {}

    def on_start(self, req):
        self.live[req.n] = self.screen.acquire(blocking=False)
        if self.live[req.n]:
            print(f"\n{BOLD}{CYAN}━━ #{req.n} {req.model} ━━ {req.title}{RESET}", flush=True)

    def on_delta(self, req, kind, text):
        if not self.live.get(req.n):
            return
        if self.mode.get(req.n) != kind:
            sys.stdout.write(f"{DIM}[thinking] " if kind == "reasoning" else f"{RESET}\n{BOLD}[answer]{RESET} ")
            self.mode[req.n] = kind
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_done(self, req):
        toks = req.tokens if req.tokens is not None else "?"
        if req.error:
            print(f"{RED}[#{req.n}] {req.error}{RESET}", flush=True)
        if self.live.pop(req.n, False):
            print(f"{RESET}\n{DIM}── #{req.n} done: {toks} tokens, {req.secs:.0f}s, finish={req.finish}{RESET}", flush=True)
            self.screen.release()
        else:
            print(f"{DIM}── #{req.n} (saved only) {toks} tokens, {req.secs:.0f}s: {req.title[:60]}{RESET}", flush=True)
        self.mode.pop(req.n, None)


def upstream_conn(base, timeout):
    u = urllib.parse.urlparse(base)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


class Proxy:
    """OpenAI-compatible streaming proxy. start() runs it in a background thread."""

    def __init__(self, upstream, port, log_dir="logs/live", timeout=3600, listener=None, host="127.0.0.1"):
        self.upstream, self.port, self.log_dir, self.timeout = upstream, port, log_dir, timeout
        self.listener = listener or Listener()
        self.counter = itertools.count(1)
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                proxy._relay(self, "GET")

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                req = json.loads(raw) if self.path.endswith("/chat/completions") else None
                if req is None or req.get("stream"):  # not a chat call, or caller streams itself
                    return proxy._relay(self, "POST", raw)
                proxy._chat(self, req)

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                err = sys.exc_info()[1]
                if isinstance(err, (BrokenPipeError, ConnectionResetError)):
                    return  # the client went away (e.g. it was cancelled); nothing to do
                proxy._notify("on_error", f"request from {client_address[0]} failed: {err!r}")

        self.server = Server((host, port), Handler)
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self):
        self.server.shutdown()
        self.server.server_close()

    def serve_forever(self):
        self.server.serve_forever()

    def _notify(self, name, *args):
        try:
            getattr(self.listener, name)(*args)
        except Exception as e:
            if name != "on_error":
                self._notify("on_error", f"listener {name} failed: {e!r}")

    def _relay(self, h, method, body=None):
        conn, prefix = upstream_conn(self.upstream, self.timeout)
        headers = {k: v for k, v in h.headers.items() if k.lower() in ("content-type", "authorization")}
        conn.request(method, prefix + h.path, body=body, headers=headers)
        r = conn.getresponse()
        data = r.read()
        h.send_response(r.status)
        h.send_header("Content-Type", r.getheader("Content-Type", "application/json"))
        h.send_header("Content-Length", str(len(data)))
        h.end_headers()
        h.wfile.write(data)

    def _chat(self, h, body):
        req = Request(next(self.counter), body.get("model"), body.get("messages", []))
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
        conn, prefix = upstream_conn(self.upstream, self.timeout)
        conn.request("POST", prefix + h.path, body=json.dumps(body),
                     headers={"Content-Type": "application/json",
                              "Authorization": h.headers.get("Authorization", "Bearer x")})
        r = conn.getresponse()
        if r.status != 200:
            data = r.read()
            req.error = f"upstream error {r.status}: {data[:300]!r}"
            self._notify("on_start", req)
            self._notify("on_done", req)
            h.send_response(r.status)
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(data)))
            h.end_headers()
            h.wfile.write(data)
            return

        self._notify("on_start", req)
        try:
            for line in r:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                req.usage = chunk.get("usage") or req.usage
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {})
                    req.finish = ch.get("finish_reason") or req.finish
                    think = delta.get("reasoning_content") or delta.get("reasoning")
                    if think:
                        req.reasoning.append(think)
                        self._notify("on_delta", req, "reasoning", think)
                    if delta.get("content"):
                        req.answer.append(delta["content"])
                        self._notify("on_delta", req, "answer", delta["content"])
        except Exception as e:  # upstream died mid-stream; still report what we have
            req.error = f"stream error: {e}"
            raise
        finally:
            self._save(req)
            self._notify("on_done", req)

        message = {"role": "assistant", "content": "".join(req.answer)}
        if req.reasoning:
            message["reasoning_content"] = "".join(req.reasoning)
        out = json.dumps({
            "id": f"proxy-{req.n}", "object": "chat.completion", "created": int(req.started),
            "model": req.model,
            "choices": [{"index": 0, "message": message, "finish_reason": req.finish}],
            "usage": req.usage or {},
        }).encode()
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.send_header("Content-Length", str(len(out)))
        h.end_headers()
        h.wfile.write(out)

    def _save(self, req):
        os.makedirs(self.log_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%H%M%S")
        toks = req.tokens if req.tokens is not None else "?"
        with open(os.path.join(self.log_dir, f"{req.n:04d}_{stamp}.txt"), "w") as f:
            f.write(f"MODEL: {req.model}\nPROMPT: {req.text}\n\n=== REASONING ===\n{''.join(req.reasoning)}\n\n"
                    f"=== ANSWER ===\n{''.join(req.answer)}\n\n=== {toks} tokens, {req.secs:.1f}s, "
                    f"finish={req.finish} ===\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="tuieval watch", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default="http://localhost:8080", help="model server base URL (no /v1)")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 for other machines)")
    p.add_argument("--log-dir", default="logs/live")
    p.add_argument("--timeout", type=float, default=3600, help="seconds to wait on the model server")
    args = p.parse_args(argv)
    print(f"Watching {args.upstream} -> http://{args.host}:{args.port}/v1  (full logs: {args.log_dir}/)", flush=True)
    Proxy(args.upstream, args.port, args.log_dir, args.timeout, PrintListener(), host=args.host).serve_forever()


if __name__ == "__main__":
    main()
