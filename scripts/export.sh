#!/usr/bin/env bash
# Prepare the TP=4 ring package: ignition join on ranks 1-3, ignition export on rank 0
# (60-90 min at 7f534bb6, most of it kernel tuning; the first export of a new pin takes about
# 2.5 h because every kernel compiles). The export publishes an entry
# in rank 0's model cache, ~/.cache/flare/packages/models/<model>/<version>/model.safetensors,
# and the log ends with its path; bin/launch-rank serves it in place. Every process is
# detached (setsid nohup) with its own log in ~/logs, so stopping this script does not
# stop the export.
# Needs: scripts/build-flare.sh on all nodes and scripts/fetch.sh on rank 0. Needs the
# GPUs: no other model may run on the four Sparks.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
read -r -a hosts <<< "$SPARK_HOSTS"
head="$SPARK_HEAD"
# Request slots (and the captured decode/verification graphs) are fixed at export; `flare run
# --max-concurrency` can only go lower.
CONCURRENCY="${CONCURRENCY:-6}"
DRAFTS="${DRAFTS:-5}"
PORT="${RENDEZVOUS_PORT:-29630}"
# 8192-token prefill chunks halve the prefill launches against 4096 (+18% cold / +26% warm on an
# 82K prompt at 7f534bb6). They need patches/0012: without it every prompt above 7,168 tokens
# faults (Xid 31) on all ranks (Hyperflare issue #266). Check each export with tests/check-unmapped.py.
CHUNK="${CHUNK:-8192}"
# csf compresses the MXFP4 expert scales, as upstream exports DeepSeek V4.1 at TP4.
EXPORT_FLAGS="${EXPORT_FLAGS:---expert-scale-compression csf}"
ADDR="$SPARK_HEAD_FABRIC"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
bin="PATH=$FLARE_BIN:\$PATH"

ssh -n "$head" "set -e; mkdir -p ~/logs
  test -f $DSV41F_MODEL_DIR/model.safetensors.index.json"
ssh -n "$head" "cd $FLARE_SRC && ($bin setsid nohup .venv/bin/ignition export \
  $DSV41F_MODEL_DIR --gpus 0 --ranks 4 --rendezvous $ADDR:$PORT --roce-ring \
  --transfer-streams 8 --max-concurrency $CONCURRENCY --mtp-draft-tokens $DRAFTS --prefill-chunk-size $CHUNK $EXPORT_FLAGS \
  --verbose > ~/logs/export-$stamp.log 2>&1 < /dev/null &); echo export started"
sleep 10
for rank in 1 2 3; do
  h="${hosts[$rank]}"
  ssh -n "$h" "mkdir -p ~/logs; cd $FLARE_SRC && ($bin setsid nohup .venv/bin/ignition join $ADDR:$PORT \
    --rank $rank --ranks 4 > ~/logs/join-$stamp.log 2>&1 < /dev/null &); echo rank $rank joined from \$(hostname)"
done
echo "logs: ~/logs/{export,join}-$stamp.log on each node; the export log on $head ends with"
echo "\"Exported cache entry: <dir>\". Then run tests/check-unmapped.py on it, and scripts/publisher.sh."
