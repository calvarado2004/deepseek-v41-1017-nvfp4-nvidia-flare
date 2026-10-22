#!/usr/bin/env python3
"""Smoke test for the served card: one short request per feature, one at a time.

    python3 -I tests/smoke.py [URL] [MODEL]
    (defaults: https://127.0.0.1:8888, DeepSeek-V4.1-Flash-1017-NVFP4-Flare)

  models     the model is listed
  count      thinking off (chat_template_kwargs.enable_thinking=false): 1..30 exact
  thinking   default mode, reasoning_effort low: reasoning, then the right answer
  tools      tool_choice auto: one get_weather call with valid JSON arguments
  forced     tool_choice required, no thinking request (patch 0004; was HTTP 400)
  video      a red dot moves left to right: the model names the direction (the TLS
             front turns video_url into frames; skipped without OpenCV)

Standard library only, plus OpenCV for the video check. Exit code 1 if a check fails.
"""
import base64, json, os, ssl, sys, tempfile, urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "https://127.0.0.1:8888").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "DeepSeek-V4.1-Flash-1017-NVFP4-Flare"
TLS = ssl._create_unverified_context()  # the certificate names the public host, not 127.0.0.1
OFF = {"chat_template_kwargs": {"enable_thinking": False}}
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


def call(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600, context=TLS) as response:
        return json.load(response)


def chat(content, **extra):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "max_tokens": 2048, **extra}
    return call("/v1/chat/completions", body)["choices"][0]


def models():
    names = [m["id"] for m in call("/v1/models")["data"]]
    return MODEL in names, names


def count():
    text = chat("Count from 1 to 30, separated by commas. Output only the numbers.", **OFF)["message"]["content"]
    return [n.strip() for n in text.strip().split(",")] == [str(i) for i in range(1, 31)], text[:120]


def thinking():
    m = chat("A train leaves at 14:35 and arrives at 17:10. How many minutes is the trip? "
             "Answer with the number only.", reasoning_effort="low")["message"]
    return bool(m.get("reasoning_content")) and "155" in (m.get("content") or ""), \
        f"reasoning {len(m.get('reasoning_content') or '')} chars, answer {m.get('content')!r}"


def tools(choice):
    c = chat("What is the weather in Lisbon right now?", tools=[WEATHER], tool_choice=choice)
    calls = c["message"].get("tool_calls") or []
    ok = len(calls) == 1 and calls[0]["function"]["name"] == "get_weather" \
        and "lisbon" in json.loads(calls[0]["function"]["arguments"])["city"].lower()
    return ok, [x["function"] for x in calls] or c["message"].get("content", "")[:120]


def video():
    try:
        import cv2, numpy as np
    except ImportError:
        return None, "OpenCV not installed"
    path = os.path.join(tempfile.mkdtemp(), "dot.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 180))
    for i in range(40):
        frame = np.full((180, 320, 3), 255, np.uint8)
        cv2.circle(frame, (20 + i * 7, 90), 14, (0, 0, 255), -1)
        out.write(frame)
    out.release()
    clip = base64.b64encode(open(path, "rb").read()).decode()
    content = [{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{clip}", "num_frames": 6}},
               {"type": "text", "text": "In which direction does the red dot move: left-to-right or "
                                        "right-to-left? Answer with that phrase only."}]
    text = chat(content, **OFF)["message"]["content"] or ""
    return "left-to-right" in text.lower().replace(" to ", "-to-"), text[:120]


CHECKS = [("models", models), ("count", count), ("thinking", thinking),
          ("tools", lambda: tools("auto")), ("forced", lambda: tools("required")), ("video", video)]

if __name__ == "__main__":
    failed = 0
    for name, check in CHECKS:
        try:
            ok, detail = check()
        except Exception as e:  # HTTP errors carry the server's message
            body = e.read().decode()[:200] if hasattr(e, "read") else ""
            ok, detail = False, f"{type(e).__name__}: {e} {body}"
        verdict = "SKIP" if ok is None else "PASS" if ok else "FAIL"
        failed += verdict == "FAIL"
        print(f"{verdict} {name:9} {detail}", flush=True)
    sys.exit(1 if failed else 0)
