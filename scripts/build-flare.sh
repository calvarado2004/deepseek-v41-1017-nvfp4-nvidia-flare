#!/usr/bin/env bash
# Build patched Flare (native CLI + Ignition venv) on one Spark. Run on the Spark itself:
#   scripts/build-flare.sh            # uses pins.env next to scripts/
# Installs build packages and cuda-toolkit-13-3 if missing, rustup and uv in ~/.
# Starts nothing.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$here/pins.env"

pkgs=(build-essential cmake ninja-build nasm pkg-config protobuf-compiler libssl-dev python3-dev
      liburing-dev libibverbs-dev git curl)
[ -x "$FLARE_CUDA/bin/nvcc" ] || pkgs+=(cuda-toolkit-13-3)
missing=()
for p in "${pkgs[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if [ ${#missing[@]} -gt 0 ]; then
  echo "apt: ${missing[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${missing[@]}"
fi

command -v cargo >/dev/null || [ -x ~/.cargo/bin/cargo ] ||
  curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
command -v uv >/dev/null || [ -x ~/.local/bin/uv ] ||
  curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$FLARE_CUDA/bin:$PATH"
export CUDA_HOME="$FLARE_CUDA"

sudo mkdir -p "$FLARE_HOME" && sudo chown "$(id -un):$(id -gn)" "$FLARE_HOME"
if [ ! -d "$FLARE_SRC/.git" ]; then
  git clone -q "$FLARE_REPO" "$FLARE_SRC"
fi
cd "$FLARE_SRC"
git fetch -q origin
git checkout -q -B card "$FLARE_COMMIT"
git -c user.name=card -c user.email=card@localhost am -q "$here"/patches/*.patch
git log --oneline -4

uv sync --locked --extra prepare --extra test
cargo build --release --manifest-path flare/Cargo.toml
# The extension embeds Helios; rebuild it so it carries the patched Helios.
uv pip install --python .venv/bin/python --no-deps --reinstall ./heliopause

mkdir -p ~/.config/flare
cfg=~/.config/flare/config.toml
if [ ! -f "$cfg" ]; then
  cat > "$cfg" <<EOF
[paths]
cuda = "$FLARE_CUDA"

[memory]
# GB10 shares DRAM with the OS; keep room for host activity and the driver.
reserve = "6G"
EOF
fi
./flare/target/release/flare --version
.venv/bin/python -c "import ignition, flashinfer, torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_capability())"
echo built
