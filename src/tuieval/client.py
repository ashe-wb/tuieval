"""Streaming client for OpenAI-compatible chat servers (llama.cpp, vLLM, LM Studio, Ollama, OpenRouter, …).

stream_chat() sends one request, reports reasoning/answer tokens as they arrive, and returns the
answer together with the numbers that matter for choosing a local model:

  ttft_s         time to the first generated token (reasoning or answer)
  total_s        wall time for the whole request
  gen_tps        generation speed, tokens/s (server-reported when available)
  prompt_tps     prompt processing speed, tokens/s (llama.cpp reports it)
  prompt_tokens, completion_tokens, and — when the split can be known — reasoning_tokens and
  answer_tokens (tokens_estimated=True means the split was estimated from text length)
"""
import http.client
import json
import socket
import time
import urllib.parse

SERVER_ERROR_PREFIXES = ("server returned HTTP 5", "server returned HTTP 429", "server error", "connection error")


def is_server_error(error):
    """True for an error that is the server's fault rather than the model's answer: HTTP 5xx or
    429, a dropped connection, a mid-stream error. A read timeout is not one: that's the model
    taking too long, and it counts against it."""
    return bool(error) and error.startswith(SERVER_ERROR_PREFIXES) and "timed out" not in error.lower() \
        and "timeout" not in error.lower()


def server_error_row(rec):
    """A stored answer that was really a server failure (never a judgement of the model)."""
    return not rec.get("pass") and is_server_error(rec.get("reason"))


class Cancelled(Exception):
    pass


class Stream:
    """One in-flight request. close() from another thread aborts it."""

    def __init__(self):
        self.conn = None
        self.sock = None     # kept separately: http.client drops conn.sock once a response streams
        self.cancelled = False

    def close(self):
        self.cancelled = True
        sock = self.sock or (self.conn.sock if self.conn is not None else None)
        if sock is not None:
            try:
                # shutdown() wakes a thread blocked reading this socket; close() alone doesn't on macOS
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass


def _conn(base_url, timeout):
    u = urllib.parse.urlparse(base_url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u.path.rstrip("/")


def post_json(base_url, path, payload, timeout=30):
    conn, prefix = _conn(base_url, timeout)
    try:
        conn.request("POST", prefix + path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = r.read()
        if r.status != 200:
            raise OSError(f"HTTP {r.status}")
        return json.loads(data)
    finally:
        conn.close()


def count_tokens(base_url, text):
    """Exact token count via llama.cpp's /tokenize; None if the server doesn't support it."""
    if not text:
        return 0
    try:
        root = base_url[:-3] if base_url.rstrip("/").endswith("/v1") else base_url
        return len(post_json(root, "/tokenize", {"content": text}, timeout=10).get("tokens", []))
    except (OSError, ValueError, http.client.HTTPException):
        return None


def stream_chat(base_url, body, on_delta=None, timeout=1800, stream=None, headers=None):
    """POST {base_url}/v1/chat/completions with streaming. Returns a dict (see module docstring).

    on_delta(kind, text) is called for "reasoning" and "answer" text as it arrives.
    """
    stream = stream or Stream()
    body = dict(body, stream=True, stream_options={"include_usage": True})
    conn, prefix = _conn(base_url, timeout)
    stream.conn = conn
    out = {"answer": "", "reasoning": "", "tool_calls": [], "finish": None, "usage": {}, "timings": {},
           "ttft_s": None, "total_s": None, "error": None}
    reasoning, answer, tools = [], [], {}
    start = time.time()
    try:
        conn.request("POST", prefix + "/v1/chat/completions", body=json.dumps(body),
                     headers={"Content-Type": "application/json", "Authorization": "Bearer none", **(headers or {})})
        stream.sock = conn.sock
        if stream.cancelled:          # cancelled while connecting
            raise OSError("cancelled")
        r = conn.getresponse()
        if r.status != 200:
            out["error"] = f"server returned HTTP {r.status}: {r.read()[:300].decode('utf-8', 'replace')}"
            return out
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("error"):        # OpenRouter reports mid-stream failures this way
                err = chunk["error"]
                out["error"] = f"server error: {err.get('message', err) if isinstance(err, dict) else err}"
                break
            out["provider"] = chunk.get("provider") or out.get("provider")   # OpenRouter: who answered
            out["usage"] = chunk.get("usage") or out["usage"]
            out["timings"] = chunk.get("timings") or out["timings"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta") or {}
                out["finish"] = ch.get("finish_reason") or out["finish"]
                think = delta.get("reasoning_content") or delta.get("reasoning")
                text = delta.get("content")
                for tc in delta.get("tool_calls") or []:
                    slot = tools.setdefault(tc.get("index", len(tools)), {"name": "", "arguments": ""})
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["arguments"] += fn.get("arguments") or ""
                if (think or text or delta.get("tool_calls")) and out["ttft_s"] is None:
                    out["ttft_s"] = time.time() - start
                if think:
                    reasoning.append(think)
                    on_delta and on_delta("reasoning", think)
                if text:
                    answer.append(text)
                    on_delta and on_delta("answer", text)
    except (OSError, http.client.HTTPException, ValueError) as e:
        if stream.cancelled:
            raise Cancelled() from None
        out["error"] = f"connection error: {e!r}"
    finally:
        conn.close()
        out["total_s"] = time.time() - start
        out["answer"], out["reasoning"] = "".join(answer), "".join(reasoning)
        for slot in (tools[k] for k in sorted(tools)):
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"].strip() else {}
            except ValueError:
                args = {"_raw": slot["arguments"]}
            out["tool_calls"].append({"name": slot["name"], "arguments": args})
    return out


def metrics(res, base_url=None):
    """Token and speed numbers for a finished request (tokenizes via the server when it can)."""
    usage, timings = res.get("usage") or {}, res.get("timings") or {}
    completion = usage.get("completion_tokens") or timings.get("predicted_n")
    prompt = usage.get("prompt_tokens") or timings.get("prompt_n")
    details = usage.get("completion_tokens_details") or {}
    reasoning_tokens, answer_tokens, estimated = details.get("reasoning_tokens"), None, False
    if completion and reasoning_tokens is None and res.get("reasoning"):
        answer_tokens = count_tokens(base_url, res["answer"]) if base_url else None
        if answer_tokens is not None:
            reasoning_tokens = max(0, completion - answer_tokens)
        else:  # split by text length
            total_chars = len(res["reasoning"]) + len(res["answer"]) or 1
            reasoning_tokens = round(completion * len(res["reasoning"]) / total_chars)
            estimated = True
    if completion is not None:
        reasoning_tokens = reasoning_tokens or 0
        answer_tokens = completion - reasoning_tokens
    # prompt tokens the server reused from earlier requests instead of computing (should be 0 in evals)
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or timings.get("cache_n")
    gen_tps = timings.get("predicted_per_second")
    if not gen_tps and completion and res.get("ttft_s") is not None and res["total_s"] - res["ttft_s"] > 0.05:
        gen_tps = completion / (res["total_s"] - res["ttft_s"])
    hosted = {k: v for k, v in (("provider", res.get("provider")), ("cost", usage.get("cost"))) if v is not None}
    return {
        **hosted,
        "prompt_tokens": prompt, "cached_tokens": cached, "completion_tokens": completion,
        "reasoning_tokens": reasoning_tokens, "answer_tokens": answer_tokens, "tokens_estimated": estimated,
        "ttft_s": res.get("ttft_s"), "total_s": res.get("total_s"),
        "gen_tps": round(gen_tps, 2) if gen_tps else None,
        "prompt_tps": round(timings["prompt_per_second"], 1) if timings.get("prompt_per_second") else None,
    }
