#!/usr/bin/env bash
# Copy the public profile of rank 0's local development key to every node, once after
# the first export. The export signs the package with that key
# (~/.config/flare/development-signing-key.pem on rank 0); every rank verifies the
# package with the profile in ~/.cache/flare/publishers/<fingerprint>.json, where
# bin/launch-rank finds it and approves that exact fingerprint
# (docs/cli-design.md, signing and local development). Needs no GPU.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/cluster.env"
read -r -a hosts <<< "$SPARK_HOSTS"
head="$SPARK_HEAD"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT

scp -q "$head:.config/flare/development-publisher.json" "$tmp/profile.json" || {
  echo "no ~/.config/flare/development-publisher.json on $head: run scripts/export.sh first" >&2; exit 1; }
# The fingerprint is the SHA-256 of the 32 public-key bytes.
fingerprint="$(python3 -I -c '
import hashlib, json, sys
profiles = json.load(open(sys.argv[1]))
if len(profiles) != 1:
    sys.exit(f"expected one development profile, found {len(profiles)}")
print(hashlib.sha256(bytes(profiles[0]["public_key"])).hexdigest())' "$tmp/profile.json")"
echo "development key $fingerprint"
for h in "${hosts[@]}"; do
  ssh -n "$h" "mkdir -p ~/.cache/flare/publishers"
  scp -q "$tmp/profile.json" "$h:.cache/flare/publishers/$fingerprint.json"
  echo "== $h: $(ssh -n "$h" "ls ~/.cache/flare/publishers/ | wc -l") profile(s)"
done
echo "done; stage and start the ranks (README, quick start)"
