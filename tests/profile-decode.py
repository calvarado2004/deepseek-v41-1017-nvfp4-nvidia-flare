# Per-token latency budget of one decode request on the Flare card. Run ON THE HEAD with
# the card otherwise idle:  python3 profile-decode.py OUTDIR [count code prose]
# Each workload: the bench.py request (temperature 0, 400 tokens, effort low, streamed)
# sent alone to 127.0.0.1:8001, /metrics before and after, the server emit time of every
# token (token_timestamps_ns). One speculative round emits its tokens together, so time
# bursts are rounds: wall time per round minus GPU time per round (verify + draft graphs,
# from /metrics) is the time outside the GPU graphs.
import json, os, re, sys, time, urllib.request
URL = "http://127.0.0.1:8001"
MODEL = os.environ.get("AB_MODEL", "DeepSeek-V4.1-Flash-1017-NVFP4-Flare")
P = {"count": "Count from 1 to 100, separated by commas. Output only the numbers.",
     "code": "Write a Python LRU cache class with get and put in O(1), with type hints. Code only.",
     "prose": "Write a 350-word essay on the history of the printing press."}
KEEP = re.compile(r"^flare_(speculative_(rounds|gpu_nanoseconds|emitted_tokens|offered_drafts|accepted_drafts)|"
                  r"proposal_(rounds|gpu_nanoseconds)|replay_(gpu_nanoseconds|readiness_wait_nanoseconds|readiness_samples)|"
                  r"engram_(gathers|gather_nanoseconds|disk_wait_nanoseconds|row_cache_hits|row_cache_misses)|"
                  r"generated_tokens|graph_launches)_total")


def metrics():
    out = {}
    for line in urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode().splitlines():
        name = line.split("{")[0].split(" ")[0]
        if KEEP.match(name):
            key, value = line.rsplit(" ", 1)
            out[key] = float(value)
    return out


def run(kind):
    body = {"model": MODEL, "messages": [{"role": "user", "content": P[kind]}], "max_tokens": 400,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": "low"},
            "return_token_ids": True, "return_token_timestamps": True}
    before = metrics()
    chunks, t0 = [], time.monotonic()
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data:") and line != "data: [DONE]":
                chunks.append((time.monotonic() - t0, json.loads(line[5:])))
    after = metrics()
    delta = {k: after[k] - before.get(k, 0.0) for k in after if after[k] - before.get(k, 0.0)}
    return chunks, delta


def summarize(kind, chunks, delta):
    def total(prefix):
        return sum(v for k, v in delta.items() if k.startswith(prefix))
    # Server timestamps (token_timestamps_ns): one per token; a round's tokens share a time.
    stamps = []
    for _, d in chunks:
        for c in d.get("choices") or []:
            stamps += [ns / 1e9 for ns in c.get("token_timestamps_ns") or []]
    first = chunks[0][0] if chunks else 0.0
    for t, d in chunks:
        if any(c.get("token_ids") for c in d.get("choices") or []):
            first = t
            break
    decode = stamps[1:]
    span = (decode[-1] - stamps[0]) if decode else 0.0
    tokens = len(decode)
    # Bursts: tokens less than 0.5 ms apart belong to one round.
    bursts, last = [], None
    for t in decode:
        if last is not None and t - last < 0.0005:
            bursts[-1][1] += 1
        else:
            bursts.append([t, 1])
        last = t
    gaps = sorted(b[0] - a[0] for a, b in zip(bursts, bursts[1:]))
    rounds = total("flare_speculative_rounds_total")
    verify = total("flare_speculative_gpu_nanoseconds_total") / 1e6
    draft = total("flare_proposal_gpu_nanoseconds_total") / 1e6
    ready = total("flare_replay_readiness_wait_nanoseconds_total") / 1e6
    hits, misses = total("flare_engram_row_cache_hits_total"), total("flare_engram_row_cache_misses_total")
    offered, accepted = total("flare_speculative_offered_drafts_total"), total("flare_speculative_accepted_drafts_total")
    wall_round = span * 1e3 / max(len(bursts) - 1, 1)
    print(f"== {kind}: {tokens} decode tokens in {span:.2f} s = {tokens / max(span, 1e-9):.1f} tok/s, TTFT {first:.2f} s")
    print(f"   client bursts {len(bursts)}  tokens/burst {tokens / max(len(bursts), 1):.2f}  "
          f"gap ms p10/p50/p90 {gaps[len(gaps) // 10]:.1f}/{gaps[len(gaps) // 2]:.1f}/{gaps[9 * len(gaps) // 10]:.1f}" if gaps else "   no bursts")
    print(f"   server rounds {int(rounds)}  verify GPU {verify / max(rounds, 1):.1f} ms/round  "
          f"draft GPU {draft / max(rounds, 1):.1f} ms/round  readiness wait {ready / max(rounds, 1):.1f} ms/round")
    print(f"   acceptance {100 * accepted / max(offered, 1):.1f}% ({int(accepted)}/{int(offered)})  "
          f"engram cache hit {100 * hits / max(hits + misses, 1):.1f}%")
    outside = wall_round - (verify + draft) / max(rounds, 1)
    print(f"   per round: wall {wall_round:.1f} ms = GPU {(verify + draft) / max(rounds, 1):.1f} ms + outside GPU {outside:.1f} ms")
    return {"kind": kind, "tokens": tokens, "span_s": span, "ttft_s": first, "bursts": len(bursts),
            "gaps_ms": [round(g * 1e3, 2) for g in gaps], "delta": delta}


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    for kind in sys.argv[2:] or ["count", "code", "prose"]:
        chunks, delta = run(kind)
        json.dump({"chunks": chunks, "delta": delta}, open(os.path.join(out, f"{kind}-raw.json"), "w"))
        json.dump(summarize(kind, chunks, delta), open(os.path.join(out, f"{kind}.json"), "w"), indent=1)
        sys.stdout.flush()
