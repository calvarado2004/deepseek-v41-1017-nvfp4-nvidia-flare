<p align="center">
  <img src="assets/deepseek-logo.svg" alt="DeepSeek" width="360">
</p>

# DeepSeek-V4.1-Flash-1017 on NVIDIA Flare, four DGX Sparks

A recipe that serves DeepSeek's official weights,
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(MXFP4 routed experts, FP8 Engram tables, loaded as published), with [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) at tensor
parallelism 4 on four NVIDIA DGX Sparks (GB10) cabled as a **switchless QSFP
ring**. Ignition prepares a signed runtime package once; the Helios runtime then
runs one rank per Spark, with RoCEnante collectives on the ring cables. 1M-token
context, thinking on by default, tool calls, images, and video (as frames). The
API is OpenAI-compatible.

The same model on the same four Sparks under vLLM is
[calvarado2004/deepseek-v41-1017-nvfp4](https://github.com/calvarado2004/deepseek-v41-1017-nvfp4)
(LibertAIDAI's NVFP4 transcode of these weights).
Flare is faster for one request; vLLM is faster for four.

| Measured, one stream unless noted (tok/s) | Flare, official weights (2026-10-25) | Flare, LibertAIDAI NVFP4 (2026-10-22) | vLLM recipe |
|---|---:|---:|---:|
| Count to 100 (median of 3; runs) | 81.9 (73.2 / 81.9 / 115.8) | 98.1 | 81 |
| Code (median of 3; runs) | 85.1 (90.0 / 85.1 / 70.8) | 89.3 | 75 |
| Prose, English (median of 3; runs) | 46.2 (46.5 / 45.0 / 46.2) | 41.6 | 32 |
| Prose, Spanish (median of 3; runs) | 47.7 (47.7 / 44.6 / 48.7) | 43.9 | – |
| Two prose streams, aggregate | 84.7 (42.3 + 42.4) | 66.8 | – |
| `code_bench`, 1 stream | 78.5 | 73.8 | 61.2 |
| `code_bench`, 4 streams, aggregate | 92.3 (23.1 per stream) | 84.4 | 126.4 |
| Cold prefill, 20K tokens (tok/s) | 3,733 / 1,416 (two runs) | 3,344 | ~3,500 |

`bench.py` rows: 350-word answers at temperature 0, low effort, 3 runs. Same
settings in both Flare columns: adaptive DSpark verification, the Engram row
cache at 6G, and the rendezvous on the fabric (see "Three settings that decide
the speed"). The two cold-prefill runs on the official package disagree; a
third measurement is open.

## Real work: one prompt, one shot

<p align="center">
  <img src="assets/neon-city-demo.gif" alt="Neon city, generated in one shot" width="760">
</p>

A single request: a long spec for a **futuristic neon city at night in the rain**
in three.js (thousands of lit windows, holographic billboards, wet reflective
streets, rain, flying-car lanes, fog, and an exact 60-second camera
flythrough), reasoning effort **Max**. The model wrote a 2,274-line page that
runs as is. [`examples/neon-city/`](examples/neon-city/) has the untouched HTML
and a 63-second recording.

| One request, 651-token prompt | |
|---|---:|
| Generated | **114,645 tokens in 30.6 min — 62.5 tok/s average** |
| First token | 841 ms |

<p align="center">
  <img src="assets/chat-neon-city.jpg" alt="The chat with the run's statistics" width="49%">
  <img src="assets/dashboard-code.jpg" alt="Flare metrics while the model writes code: 91.4 tok/s, 5.08 tokens per step" width="49%">
</p>

The dashboard (right, from a code request on the same card) shows what DSpark
does on code: 5.08 tokens per step and 96–100% acceptance on the first three
draft positions. Free reasoning at temperature 1.0 drafts less well, so the
average over a long Max-effort request is lower than the code-phase peak.

## Quality

| Check (official weights, 2026-10-25) | Result |
|---|---|
| Count to 100, 3 runs | exact 3/3 |
| Deduction puzzle | correct ("dog"), 4,548 tokens, no loops, no foreign script |
| Tool calls, listed and deferred, stream and non-stream | pass |
| Needle in a haystack at 81,920 / 262,144 / 524,288 / 955,000 tokens | found at every length |
| Images, the vLLM recipe's 44-image set (count, OCR, color, chart, spatial, multi-image) | **44/44** |
| Video, a moving object (TLS front frames) | pass |
| Finished reply while another request decodes (`scripts/check_response_completion.py`, patch 0005) | pass: the short reply closed 2 ms after its last token while an 8,192-token reply kept decoding |
| `tests/smoke.py` (6 checks) | 6/6 |
| Qwen Code, a small car-dealership Python project (LibertAIDAI weights, 2026-10-22) | 18 tool calls, 6 modules, **36/36 tests pass** |
| Long agent session, 401 messages (patch 0006, LibertAIDAI weights) | first token in 0.3 s with the prefix cache; cold 69.7 s for 41.7K tokens |

## Hardware

- 4 × DGX Spark (GB10, 128 GB unified memory). Rank 0 needs about 1 TB free
  disk: the checkpoint (475 GiB) and the package (474 GiB). Ranks 1–3 need
  ~119 GiB each for their rank of the package.
- Four QSFP cables in a ring: rank 0 → 1 → 2 → 3 → 0, each cable on its own
  /24. The vLLM recipe's README describes the cabling and addresses we use.
- Ubuntu with CUDA 13.3 (`scripts/build-flare.sh` installs `cuda-toolkit-13-3`
  if missing), and one account (`SPARK_USER`) with passwordless sudo and SSH
  between the nodes. No Docker.

## What the stack is

| Layer | Choice |
|---|---|
| Engine | Flare `5f37bd1` + 5 patches (`patches/`), built from source on each node |
| Package | `ignition export` at TP=4 with `--roce-ring`, max concurrency 4, 5 DSpark drafts, prefill chunk 4,096, csf expert-scale compression |
| Experts | MXFP4, as DeepSeek publishes them (Flare's stock deepseek41 recipe) |
| Engram tables | FP8 E4M3 rows, as published; read from NVMe through a 6G row cache |
| Speculative decoding | DSpark, 5 drafts, adaptive verification (package default) |
| Processes | one systemd unit per rank in Flare's documented manual mode (`docs/multi-host.md`, "Start ranks manually"); each rank logs to its own journal |
| HTTPS | `bin/tls-front.py`: TLS 1.2+ on :8888 in front of Flare's plain HTTP on 127.0.0.1:8001 |
| Video | the TLS front turns `video_url` parts into timestamped JPEG frames (`bin/dsv41_video_frames.py`, the vLLM recipe's module) |

### The patches

| Patch | Why |
|---|---|
| 0002 | Raise the dynamic shared memory limits before graph nodes are rebuilt (dbotwinick's fix on [#2](https://github.com/NVIDIA/flare-local-inference/issues/2)). Without it the load fails at "Prepare execution". |
| 0003 | Give Helios the RoCE proxy ABI 10 that the locked FlashInfer exports ([#7](https://github.com/NVIDIA/flare-local-inference/issues/7), closed upstream after our pin). |
| 0004 | A request with `tool_choice: "required"` that asks for no thinking mode is encoded in chat mode. Flare thinks by default, so these requests got HTTP 400 "thinking mode does not support required tool choice"; Qwen Code sends them for its side queries. A request that asks for thinking and forces a tool still gets the 400. |
| 0005 | Upstream [#75](https://github.com/NVIDIA/flare-local-inference/pull/75) by Martin Vit, ported to the pin (one test edit). A finished response must copy its cache snapshot before Flare sends `finish_reason` and `[DONE]`, and that copy lost to other requests' decode. Two long Qwen Code sessions at 265-325K tokens stalled: a finished `compress` reply waited 16.6 h and came back only when the other session's turn was cancelled. With the patch, `scripts/check_response_completion.py` closes a short reply 2.5 ms after its last token while a long one keeps decoding. The package does not need a new export. |
| 0006 | Find the prefix-cache boundaries of a request with one tokenization. Flare at the pin renders and tokenizes every message prefix, so the cost is messages × prompt length on one frontend thread: a ~290K-token Qwen Code turn stayed in the frontend for more than 15 minutes, with the GPU idle. The patch renders only the text of each prefix and cuts the prompt's own tokens through byte offsets; Jinja templates, and any prompt whose offsets do not reproduce its token IDs, keep the old path. Tests compare it with the old function, also with the checkpoint tokenizer. Upstream `67ba265e` (after our pin) removes the same loop. |

## Three settings that decide the speed

1. **The rendezvous must be on the fabric.** The ranks exchange a control
   message every decode round over TCP 29800 on rank 0. Our first run put it on
   Wi-Fi (17–29 ms RTT, 185 ms peaks): 37–61 ms outside the GPU each round, and
   36/33/15 tok/s instead of 98/89/42. Two places decide the route:
   `SPARK_HEAD_FABRIC` for the systemd ranks, and the agents' addresses for
   `flare orbit run` (orbit puts the rendezvous on rank 0's route toward rank 1's
   agent address). `scripts/agents.sh` pairs the agents on the fabric.
2. **Engram row cache.** Off by default: every row is a direct-I/O NVMe read (7.2 M
   reads, 4.6 ms a gather). `DSV41F_ENGRAM_CACHE=6G` keeps about 8 GiB
   MemAvailable per node.
3. **Keep adaptive verification.** `--no-mtp-adaptive-verification` was slower on
   every bench (code 78.5 vs 89.3, prose 35.6 vs 41.6) and drafts nothing at
   batch > 1.

## Quick start

1. **Edit `cluster.env`**: SSH names of the four nodes in ring order, the head,
   the account, and the fabric addresses (rank 0's, and ranks 1–3 as rank 0
   reaches them).
2. **Build Flare on every node** (run it on the node):
   `scripts/build-flare.sh`. It clones Flare at the pinned commit and applies
   `patches/`.
3. **Fetch the checkpoint** to rank 0: `scripts/fetch.sh`
   (detached; `ssh <head> tail -f logs/fetch-dsv41f.log`). Stop every model on
   the head first: on GB10 page cache is GPU memory.
4. **Pair the agents**: `scripts/agents.sh`.
5. **Export the package** (~81 min, needs all four GPUs): `scripts/export.sh`,
   then watch `~/logs/export-*.log` on the head.
6. **Distribute it once**: `scripts/distribute.sh`. Flare shows your development
   key's disclosure on first use; type `local-development` to accept it. The
   peers then fetch their ranks (~119 GiB each), and the script stops the group
   when `/health` answers.
7. **Stage**: `scripts/stage.sh` (runtime, env, unit). Put a TLS certificate at
   `/etc/ssl/flare/{fullchain,privkey}.pem` on the head, readable by
   `SPARK_USER`, or set `TLS_FRONT_CERT` / `TLS_FRONT_KEY` in the env file.
8. **Start** ranks 1–3, then rank 0:
   `sudo systemctl start spark-llm-dsv41-flare@{1,2,3}` and `…@0`. Rank 0
   reports ready when `/health` answers and a warm-up burst is done, a few
   minutes later. Logs: `journalctl -u spark-llm-dsv41-flare@N`.
9. **Test** on the head: `python3 -I tests/smoke.py`.

The API is on `https://<head>:8888/v1`, model `DeepSeek-V4.1-Flash-1017-Flare`.
`stage.sh` overwrites the env on the nodes: change `deepseek-v41-1017-flare.env`
here, not on the nodes. Re-export (step 5) and re-distribute (step 6) after a
Flare update that changes the package format.

## Requests

- Thinking is on by default. `reasoning_effort` takes `low`, `high`, `xhigh`,
  `max`; Flare rejects `none`.
- Thinking off: `"chat_template_kwargs": {"enable_thinking": false}`. Flare
  rejects template keys it does not know (vLLM's `thinking` is one).
- Flare ignores `thinking_token_budget`. The model card's sampling is
  temperature 1.0, top_p 1.0; this recipe does not force it, so send it from
  the client (at lower temperatures V4.1 can loop in its reasoning).
- Video: a `video_url` part with a `data:` URL or an http(s) URL, and optional
  `num_frames` / `fps`. Frames count as images; a bad or unreadable clip gets
  HTTP 400 from the TLS front.

## Known gaps

- **Four streams** give 92.3 tok/s in aggregate on Flare and 126.4 on vLLM: the
  package captured few verification widths above batch 1, so concurrent
  requests rarely draft. An export with wider batch-2/4 captures is the open item.
- **Prefix reuse** happens at message boundaries only, so long multi-turn
  agent sessions re-prefill more than on vLLM.
- **Literal image placeholders**: Flare rejects request text that contains
  `<｜image｜>` or `<｜deepseek_image｜>` (HTTP 400, "literal DeepSeek image
  placeholder is ambiguous"), as DeepSeek's own recipe does. An agent that reads
  a DeepSeek tokenizer, template or recipe file keeps that text in its history,
  so every later turn of that session fails; a new session works. A patch that
  encodes the literal text as plain text is in test, on a newer Flare pin.
- Flare's docs call physical ring cabling and full-model ring serving
  "unqualified". This recipe is a community result on that path, not an NVIDIA
  configuration.

## Repository layout

```
bin/launch-rank                 one rank: flare run with the recipe's settings; rank 0 adds HTTP and the TLS front
bin/tls-front.py                TLS on :8888, video parts become frames
bin/dsv41_video_frames.py       the frame sampler (from the vLLM recipe)
patches/                        0002-0006 for Flare 5f37bd1 (see NOTICE)
scripts/build-flare.sh          build patched Flare on one node
scripts/fetch.sh                the official checkpoint to rank 0
scripts/agents.sh               Flare agents, paired on the fabric
scripts/export.sh               four-rank ignition export / join
scripts/distribute.sh           one orbit run: each peer gets its rank and the publisher profile
scripts/stage.sh                bin, env and unit to the four nodes
systemd/                        the Type=notify unit, one rank per node (no [Install]: never start at boot)
tests/smoke.py                  six one-request checks
tests/profile-decode.py         per-round wall time vs GPU time of one decode
examples/neon-city/             a one-shot run: untouched output, recording
cluster.env                     your nodes
deepseek-v41-1017-flare.env     every knob, with the measured reason for its value
pins.env                        checkpoint revision, Flare commit, paths
```

## Credits

- **NVIDIA** — Flare: Ignition, Helios, RoCEnante, DSpark support.
- **dbotwinick** — the shared-memory fix on Flare issue #2 (patch 0002), and
  issue #7 (patch 0003).
- **Martin Vit** — Flare PR #75, the terminal-snapshot fix (patch 0005).
- **DeepSeek** — the model, its weights and DSpark.
- **LibertAIDAI** — the NVFP4 checkpoint that earlier versions of this recipe served.
- **FlashInfer** contributors — the RoCE proxy sources in patch 0003.

## License

The recipe's own files are MIT (`LICENSE`). The patches are changes to Flare and
keep its license (Apache-2.0); see `NOTICE`. The DeepSeek logo
(`assets/deepseek-logo.svg`, from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DeepSeek_logo.svg),
MIT, originally from deepseek-ai) is a trademark of DeepSeek. This is an
independent community recipe, not affiliated with or endorsed by DeepSeek or
NVIDIA.
