#!/usr/bin/env bash
# Fetch deepseek-ai/DeepSeek-V4.1-Flash-1017 (475.3 GiB) to rank 0. Flare's
# deepseek41 recipe loads it as published. Rank 0 only: the export sends each peer its
# part. Runs detached on the head with its own log; stopping this shell does not stop
# it. Re-run to resume.
#
# Stop every model on the head first: the download fills page cache, which is GPU
# memory on GB10.
#
#   scripts/fetch.sh
#   ssh $SPARK_HEAD tail -f logs/fetch-dsv41f.log
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
H=$SPARK_HEAD
ssh "$H" "mkdir -p ~/logs; sudo mkdir -p $DSV41F_MODEL_DIR && sudo chown $SPARK_USER:$SPARK_USER $(dirname $DSV41F_MODEL_DIR) $DSV41F_MODEL_DIR
  if pgrep -f '[h]f download $DSV41F_MODEL_ID' >/dev/null; then echo 'already running'; exit 0; fi
  (setsid nohup bash -c 'set -e
    ~/.local/bin/uvx --from huggingface_hub[hf_xet] hf download $DSV41F_MODEL_ID \
      --revision $DSV41F_MODEL_REVISION --local-dir $DSV41F_MODEL_DIR
    echo $DSV41F_MODEL_REVISION > $DSV41F_MODEL_DIR/.hf-revision
    echo fetch done' > ~/logs/fetch-dsv41f.log 2>&1 < /dev/null &)
  echo started"
echo "log: ssh $H tail -f logs/fetch-dsv41f.log"
