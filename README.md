<p align="center">
  <img src="assets/deepseek-logo.svg" alt="DeepSeek" width="360">
</p>

# DeepSeek-V4.1-Flash-1017 NVFP4 on NVIDIA Flare, four DGX Sparks

A recipe that serves
[`LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4`](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4)
with [NVIDIA Flare](https://github.com/NVIDIA/flare-local-inference) at tensor
parallelism 4 on four NVIDIA DGX Sparks (GB10) cabled as a **switchless QSFP
ring**. Ignition prepares a signed runtime package once; the Helios runtime then
runs one rank per Spark, with RoCEnante collectives on the ring cables. 1M-token
context, thinking on by default, tool calls, images, and video (as frames). The
API is OpenAI-compatible.

The same checkpoint on the same four Sparks under vLLM is
[calvarado2004/deepseek-v41-1017-nvfp4](https://github.com/calvarado2004/deepseek-v41-1017-nvfp4).
Flare is faster for one request; vLLM is faster for four.

| Measured, one stream unless noted (tok/s) | Flare | vLLM recipe |
|---|---:|---:|
| Count to 100 | **92–98** | 81 |
| Code | **85–89** | 75 |
| Prose (English / Spanish) | **42–45 / 44–48** | 32 / – |
| Two prose streams, aggregate | 65–67 | – |
| `code_bench`, 1 stream | **73.8** | 61.2 |
| `code_bench`, 4 streams, aggregate | 84.4 | **126.4** |
| Cold prefill, 20K tokens | 3,330–3,344 | ~3,500 |

Measured 2026-10-22 with adaptive DSpark verification, the Engram row cache at
6G, and the rendezvous on the fabric (see "Three settings that decide the speed").

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

| Check | Result |
|---|---|
| Count to 100, 3 runs | exact 3/3 |
| Deduction puzzle (6,006 tokens of reasoning) | correct, no loops |
| Tool calls, listed and deferred, stream and non-stream | pass |
| Needle in a haystack at 80K, 258K, 517K, 943K tokens | found at every length (943K: 483 s) |
| 80 concurrent short requests | 80/80 correct |
| Video, a moving object (TLS front frames) | direction and visibility right |
| Qwen Code, a small car-dealership Python project | 18 tool calls, 6 modules, **36/36 tests pass**; Qwen Code's side queries pass with patch 0004 |
| `tests/smoke.py` (6 checks) | 6/6 in 12 s |

Images work (`image_url` parts); the 44-image set that the vLLM recipe scores is
not scored on Flare yet.

## Hardware

- 4 × DGX Spark (GB10, 128 GB unified memory). Rank 0 needs about 1.1 TB free
  disk: the checkpoint (400 GB), its Flare view (190 GB of widened Engram
  tables) and the package (504 GB). Ranks 1–3 need ~126 GiB each for their rank
  of the package.
- Four QSFP cables in a ring: rank 0 → 1 → 2 → 3 → 0, each cable on its own
  /24. The vLLM recipe's README describes the cabling and addresses we use.
- Ubuntu with CUDA 13.3 (`scripts/build-flare.sh` installs `cuda-toolkit-13-3`
  if missing), and one account (`SPARK_USER`) with passwordless sudo and SSH
  between the nodes. No Docker.

## What the stack is

| Layer | Choice |
|---|---|
| Engine | Flare `5f37bd1` + 4 patches (`patches/`), built from source on each node |
| Package | `ignition export` at TP=4 with `--roce-ring`, max concurrency 4, 5 DSpark drafts, prefill chunk 4,096 |
| Experts | NVFP4 W4A4 (patch 0001 teaches Flare's DeepSeek-V4.1 recipe the ModelOpt NVFP4 layout) |
| Engram tables | widened from packed E2M1 to E4M3 rows, exactly (`scripts/make-flare-checkpoint.py`); rows from NVMe through a 6G row cache |
| Speculative decoding | DSpark, 5 drafts, adaptive verification (package default) |
| Processes | one systemd unit per rank in Flare's documented manual mode (`docs/multi-host.md`, "Start ranks manually"); each rank logs to its own journal |
| HTTPS | `bin/tls-front.py`: TLS 1.2+ on :8888 in front of Flare's plain HTTP on 127.0.0.1:8001 |
| Video | the TLS front turns `video_url` parts into timestamped JPEG frames (`bin/dsv41_video_frames.py`, the vLLM recipe's module) |

### The patches

| Patch | Why |
|---|---|
| 0001 | Load ModelOpt NVFP4 routed experts (W4A4) when `quantized_layers` declares them. Without it the deepseek41 recipe expects its own layout. |
| 0002 | Raise the dynamic shared memory limits before graph nodes are rebuilt (dbotwinick's fix on [#2](https://github.com/NVIDIA/flare-local-inference/issues/2)). Without it the load fails at "Prepare execution". |
| 0003 | Give Helios the RoCE proxy ABI 10 that the locked FlashInfer exports ([#7](https://github.com/NVIDIA/flare-local-inference/issues/7), closed upstream after our pin). |
| 0004 | A request with `tool_choice: "required"` that asks for no thinking mode is encoded in chat mode. Flare thinks by default, so these requests got HTTP 400 "thinking mode does not support required tool choice"; Qwen Code sends them for its side queries. A request that asks for thinking and forces a tool still gets the 400. |

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
3. **Fetch the checkpoint** to rank 0 and make the Flare view: `scripts/fetch.sh`
   (detached; `ssh <head> tail -f logs/fetch-dsv41f.log`). Stop every model on
   the head first: on GB10 page cache is GPU memory.
4. **Pair the agents**: `scripts/agents.sh`.
5. **Export the package** (~81 min, needs all four GPUs): `scripts/export.sh`,
   then watch `~/logs/export-*.log` on the head.
6. **Distribute it once**: `scripts/distribute.sh`. Flare shows your development
   key's disclosure on first use; type `local-development` to accept it. The
   peers then fetch their ranks (~126 GiB each), and the script stops the group
   when `/health` answers.
7. **Stage**: `scripts/stage.sh` (runtime, env, unit). Put a TLS certificate at
   `/etc/ssl/flare/{fullchain,privkey}.pem` on the head, readable by
   `SPARK_USER`, or set `TLS_FRONT_CERT` / `TLS_FRONT_KEY` in the env file.
8. **Start** ranks 1–3, then rank 0:
   `sudo systemctl start spark-llm-dsv41-flare@{1,2,3}` and `…@0`. Rank 0
   reports ready when `/health` answers and a warm-up burst is done, a few
   minutes later. Logs: `journalctl -u spark-llm-dsv41-flare@N`.
9. **Test** on the head: `python3 -I tests/smoke.py`.

The API is on `https://<head>:8888/v1`, model `DeepSeek-V4.1-Flash-1017-NVFP4-Flare`.
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

- **Four streams** are 33% slower than vLLM: the package captured few
  verification widths above batch 1, so concurrent requests rarely draft. An
  export with wider batch-2/4 captures is the open item.
- **Prefix reuse** happens at message boundaries only, so long multi-turn
  agent sessions re-prefill more than on vLLM.
- Flare's docs call physical ring cabling and full-model ring serving
  "unqualified". This recipe is a community result on that path, not an NVIDIA
  configuration.

## Repository layout

```
bin/launch-rank                 one rank: flare run with the recipe's settings; rank 0 adds HTTP and the TLS front
bin/tls-front.py                TLS on :8888, video parts become frames
bin/dsv41_video_frames.py       the frame sampler (from the vLLM recipe)
patches/                        0001-0004 for Flare 5f37bd1 (see NOTICE)
scripts/build-flare.sh          build patched Flare on one node
scripts/fetch.sh                checkpoint to rank 0, then make-flare-checkpoint.py
scripts/make-flare-checkpoint.py   exact E2M1 -> E4M3 widening of the Engram tables into a view directory
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
- **LibertAIDAI** — the NVFP4 checkpoint.
- **DeepSeek** — the model and DSpark.
- **FlashInfer** contributors — the RoCE proxy sources in patch 0003.

## License

The recipe's own files are MIT (`LICENSE`). The patches are changes to Flare and
keep its license (Apache-2.0); see `NOTICE`. The DeepSeek logo
(`assets/deepseek-logo.svg`, from
[Wikimedia Commons](https://commons.wikimedia.org/wiki/File:DeepSeek_logo.svg),
MIT, originally from deepseek-ai) is a trademark of DeepSeek. This is an
independent community recipe, not affiliated with or endorsed by DeepSeek or
NVIDIA.
