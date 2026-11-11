#!/usr/bin/env python3
"""Cold-prefill ladder: one request per size, unique text each time (no cached prefix).
Stops at the first failure. Usage: prefill-ladder.py WORDS [WORDS ...]
Talks to rank 0's plain HTTP on 127.0.0.1:8001."""
import json, random, sys, time, urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash-1017-Flare"
VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu").split()

for words in map(int, sys.argv[1:]):
    rnd = random.Random(time.time_ns())
    lines = []
    for i in range(words // 12):
        lines.append(f"line {i}: " + " ".join(rnd.choice(VOCAB) for _ in range(10)) + f" {rnd.randrange(10**6)}")
    doc = "\n".join(lines)
    body = {"model": MODEL, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": doc + "\nHow many lines are there? Answer briefly."}]}
    t0 = time.time()
    try:
        req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as r:
            out = json.load(r)
        u = out.get("usage", {})
        dt = time.time() - t0
        print(f"OK   words={words} prompt_tokens={u.get('prompt_tokens')} cached={((u.get('prompt_tokens_details') or {}).get('cached_tokens'))} "
              f"wall={dt:.2f}s ~{(u.get('prompt_tokens') or 0)/dt:.0f} tok/s", flush=True)
    except Exception as e:
        body = e.read().decode()[:300] if hasattr(e, "read") else ""
        print(f"FAIL words={words} after {time.time()-t0:.2f}s: {type(e).__name__}: {e} {body}", flush=True)
        sys.exit(1)
