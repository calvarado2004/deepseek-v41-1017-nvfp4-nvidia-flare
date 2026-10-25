#!/usr/bin/env bash
# Prepare the TP=4 ring package: ignition join on ranks 1-3, ignition export on rank 0
# (about 81 minutes). Every process is detached (setsid nohup) with its own log in
# ~/logs, so stopping this script does not stop the export.
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
CONCURRENCY="${CONCURRENCY:-4}"
DRAFTS="${DRAFTS:-5}"
PORT="${RENDEZVOUS_PORT:-29630}"
NAME="${PACKAGE_NAME:-dsv41-1017-official-tp4ring-c${CONCURRENCY}-dspark${DRAFTS}-csf}"
# csf compresses the MXFP4 expert scales, as upstream exports DeepSeek V4.1 at TP4.
EXPORT_FLAGS="${EXPORT_FLAGS:---expert-scale-compression csf}"
ADDR="$SPARK_HEAD_FABRIC"
stamp="$(date -u +%Y%m%dT%H%M%SZ)"
bin="PATH=$FLARE_SRC/flare/target/release:\$PATH"

ssh -n "$head" "set -e; sudo mkdir -p $DSV41F_PACKAGE_DIR; sudo chown $SPARK_USER:$SPARK_USER $DSV41F_PACKAGE_DIR; mkdir -p ~/logs
  test -f $DSV41F_MODEL_DIR/model.safetensors.index.json
  test ! -e $DSV41F_PACKAGE_DIR/$NAME.safetensors"
ssh -n "$head" "cd $FLARE_SRC && ($bin setsid nohup .venv/bin/ignition export \
  $DSV41F_MODEL_DIR --gpus 0 --ranks 4 --rendezvous $ADDR:$PORT --roce-ring --cluster ring \
  --transfer-streams 8 --max-concurrency $CONCURRENCY --mtp-draft-tokens $DRAFTS --prefill-chunk-size 4096 $EXPORT_FLAGS \
  --output $DSV41F_PACKAGE_DIR/$NAME.safetensors --verbose \
  > ~/logs/export-$stamp.log 2>&1 < /dev/null &); echo export started"
sleep 10
for rank in 1 2 3; do
  h="${hosts[$rank]}"
  ssh -n "$h" "mkdir -p ~/logs; cd $FLARE_SRC && ($bin setsid nohup .venv/bin/ignition join $ADDR:$PORT \
    --rank $rank --ranks 4 > ~/logs/join-$stamp.log 2>&1 < /dev/null &); echo rank $rank joined from \$(hostname)"
done
echo "logs: ~/logs/{export,join}-$stamp.log; package $DSV41F_PACKAGE_DIR/$NAME.safetensors on $head"
