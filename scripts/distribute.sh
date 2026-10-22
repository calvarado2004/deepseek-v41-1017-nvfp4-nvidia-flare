#!/usr/bin/env bash
# Send each peer its rank of the package and the publisher profile, once after every
# export: one `flare orbit run` (docs/orbit.md) that the agents serve, then stop it.
# The ranks land in ~/.cache/flare/packages/<export>-rank<N>/ and the key's public
# profile in ~/.cache/flare/publishers/<fingerprint>.json on every node, where
# bin/launch-rank finds them.
#
# Interactive: on first use of your development key Flare shows its disclosure and
# asks you to type `local-development` (docs/cli-design.md). That acceptance is
# yours to give; the units then pass the same fingerprint unattended.
# Needs the GPUs: stop the systemd ranks first.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
# shellcheck disable=SC1091
source "$here/deepseek-v41-1017-flare.env"
read -r -a hosts <<< "$SPARK_HOSTS"
head="$SPARK_HEAD"
flare="$FLARE_SRC/flare/target/release/flare"
package="${hosts[0]}:$DSV41F_PACKAGE_DIR/$DSV41F_PACKAGE_NAME.safetensors"

ssh -t "$head" "$flare orbit run $package --cluster ring --name dsv41-flare --gpus 0 --roce-ring --detach \
  --host 127.0.0.1 --port 8001 --served-model-name $DSV41F_SERVED_MODEL_NAME"
echo "waiting for /health on $head (the peers fetch their ranks first: ~126 GiB each)"
until ssh -n "$head" "curl -sf -m 5 -o /dev/null http://127.0.0.1:8001/health"; do sleep 30; done
ssh -n "$head" "$flare orbit stop dsv41-flare --cluster ring"
for h in "${hosts[@]}"; do
  echo "== $h"
  ssh -n "$h" "ls -d ~/.cache/flare/packages/*-rank* 2>/dev/null; ls ~/.cache/flare/publishers/"
done
echo "distributed; start the systemd ranks (README, quick start)"
