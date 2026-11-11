#!/usr/bin/env python3
"""Chunk probes against rank 0 (127.0.0.1:8001). Each case prints OK/FAIL and stops at the first FAIL.
  cold N        one unique prompt of about N tokens
  extend N M    a unique prompt of about N tokens (cached), then the same prompt plus about M tokens"""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash-1017-Flare"
VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()
TOK_PER_LINE = 24.6  # measured: 2000 words (166 lines) -> ~4,100 tokens

def text(tokens, rnd, start=0):
    return "\n".join(f"line {start+i}: " + " ".join(rnd.choice(VOCAB) for _ in range(10)) + f" {rnd.randrange(10**6)}"
                     for i in range(int(tokens / TOK_PER_LINE)))

def ask(doc, label):
    msgs = doc if isinstance(doc, list) else [{"role": "user", "content": doc}]
    body = {"model": MODEL, "max_tokens": 4, "chat_template_kwargs": {"enable_thinking": False},
            "messages": msgs}
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            u = json.load(r).get("usage", {})
        print(f"OK   {label}: prompt={u.get('prompt_tokens')} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens')} {time.time()-t0:.2f}s", flush=True)
    except Exception as e:
        b = e.read().decode()[:240] if hasattr(e, "read") else ""
        print(f"FAIL {label}: {type(e).__name__} {e} {b}", flush=True); sys.exit(1)

args = sys.argv[1:]
while args:
    rnd = random.Random(time.time_ns()); kind = args.pop(0)
    if kind == "cold":
        n = int(args.pop(0)); ask(text(n, rnd), f"cold {n}")
    elif kind == "extend":
        n, m = int(args.pop(0)), int(args.pop(0)); base = text(n, rnd)
        ask(base, f"extend base {n}"); time.sleep(1)
        ask([{"role": "user", "content": base}, {"role": "assistant", "content": "Noted."},
             {"role": "user", "content": text(m, rnd, 100000)}], f"extend {n}+{m}")
