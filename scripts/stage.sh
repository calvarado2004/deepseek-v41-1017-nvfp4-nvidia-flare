#!/usr/bin/env bash
# Stage the card on the four Sparks. Starts nothing and pulls nothing.
#   runtime  /opt/spark-llm/profiles/deepseek-v41-1017-flare/bin
#   env      /etc/spark-llm/profiles/deepseek-v41-1017-flare.env (+ DSV41F_MASTER_ADDR)
#   unit     /etc/systemd/system/spark-llm-dsv41-flare@.service (+ SPARK_USER, SPARK_RING_PEERS)
# Build (scripts/build-flare.sh), fetch, export and distribute are separate steps:
# they compile or move hundreds of GB.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
# shellcheck disable=SC1091
source "$here/deepseek-v41-1017-flare.env"
read -r -a hosts <<< "$SPARK_HOSTS"
U=spark-llm-dsv41-flare@.service
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
{ cat "$here/deepseek-v41-1017-flare.env"; echo; echo "# from cluster.env"
  echo "DSV41F_MASTER_ADDR=$SPARK_HEAD_FABRIC"; } > "$tmp/deepseek-v41-1017-flare.env"
sed -e "s/@SPARK_USER@/$SPARK_USER/g" -e "s/@SPARK_RING_PEERS@/$SPARK_RING_PEERS/g" \
  "$here/systemd/$U" > "$tmp/$U"
for i in "${!hosts[@]}"; do
  h="${hosts[$i]}"
  echo "== $h (rank $i)"
  ssh -n "$h" "sudo mkdir -p $FLARE_HOME /etc/spark-llm/profiles && sudo chown $SPARK_USER:$SPARK_USER $FLARE_HOME"
  rsync -a --delete --exclude __pycache__ "$here/bin/" "$h:$FLARE_HOME/bin/"
  if [ "$i" = 0 ]; then
    # Video frames in the TLS front need OpenCV.
    ssh -n "$h" "python3 -c 'import cv2' 2>/dev/null || sudo apt-get install -y -q python3-opencv >/dev/null"
  fi
  scp -q "$tmp/deepseek-v41-1017-flare.env" "$tmp/$U" "$h:/tmp/"
  if [ "$i" = 0 ]; then
    package="$DSV41F_PACKAGE_DIR/$DSV41F_PACKAGE_NAME.safetensors"
  else
    package="\$HOME/.cache/flare/packages/*-rank$i/$DSV41F_PACKAGE_NAME.safetensors"
  fi
  ssh -n "$h" "set -e
    sudo install -m 644 /tmp/deepseek-v41-1017-flare.env /etc/spark-llm/profiles/deepseek-v41-1017-flare.env
    sudo install -m 644 '/tmp/$U' /etc/systemd/system/
    sudo systemctl daemon-reload
    ! grep -q '^\[Install\]' '/etc/systemd/system/$U'
    [ -x $FLARE_SRC/flare/target/release/flare ] && echo '  flare present' || echo '  flare MISSING (scripts/build-flare.sh)'
    ls $package >/dev/null 2>&1 && echo '  package present' || echo '  package MISSING (scripts/export.sh, scripts/distribute.sh)'
    ls \$HOME/.cache/flare/publishers/*.json >/dev/null 2>&1 && echo '  publisher profile present' || echo '  publisher profile MISSING (scripts/distribute.sh)'
    echo staged"
done
