#!/usr/bin/env bash
# Install the Flare agent on the four Sparks and pair them with rank 0 (the controller).
# Run after scripts/build-flare.sh finished on every node. Starts no model. The agents
# are only used once, by scripts/distribute.sh, to send each peer its rank of the package.
#   agent    systemd user unit flare-agent on each node, TCP 29900 on all interfaces
#   orbit    ~/.config/flare/orbit.toml on rank 0, cluster "ring" in cable order
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"
# shellcheck disable=SC1091
source "$here/cluster.env"
read -r -a hosts <<< "$SPARK_HOSTS"
read -r -a peers <<< "$SPARK_RING_PEERS"
head="$SPARK_HEAD"
flare="$FLARE_SRC/flare/target/release/flare"
for h in "${hosts[@]}"; do
  echo "== $h"
  # --accept-first: the first controller pairs without a code (trusted management network).
  ssh -n "$h" "loginctl enable-linger \$(id -un) 2>/dev/null || sudo loginctl enable-linger \$(id -un)
    $flare agent install --accept-first | grep -iv 'code' || true
    systemctl --user is-active flare-agent"
done
ssh -n "$head" "$flare orbit pair ${hosts[*]}"
nodes="$(printf '"%s", ' "${hosts[@]}" | sed 's/, $//')"
ssh "$head" "grep -q '^\[cluster.ring\]' ~/.config/flare/orbit.toml || cat >> ~/.config/flare/orbit.toml" <<TOML

[cluster.ring]
nodes = [$nodes]
TOML
# Point the peers at their fabric addresses. orbit puts the rank-0 rendezvous on the
# route toward rank 1's agent address; a name that resolves to Wi-Fi or the LAN puts
# the per-step control stream there.
ssh "$head" "python3 -I - ${hosts[*]:1} -- ${peers[*]}" <<'PY'
import os, re, sys
args = sys.argv[1:]
cut = args.index("--")
fabric = dict(zip(args[:cut], args[cut + 1:]))
path = os.path.expanduser("~/.config/flare/orbit.toml")
blocks = re.split(r"(?m)^(?=\[)", open(path).read())
for i, block in enumerate(blocks):
    name = re.search(r'(?m)^name = "([^"]+)"', block)
    if block.startswith("[[node]]") and name:
        node = name.group(1)
        ip = fabric.get(node) or fabric.get(node.split(".")[0])
        if ip:
            blocks[i] = re.sub(r'(?m)^address = "[^"]*"', f'address = "{ip}:29900"', block)
open(path, "w").write("".join(blocks))
PY
ssh -n "$head" "grep -E '^(name|address)' ~/.config/flare/orbit.toml; $flare orbit status --cluster ring"
