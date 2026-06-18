#!/bin/bash
set -euo pipefail

VARIANT="${VARIANT:-torch212-cxx11-cu130-x86_64-linux}"
OUT_DIR="/work/kernel-hub/build-output/${VARIANT}"

echo "=== Installing apt deps ==="
apt update && apt install -y git curl xz-utils ca-certificates

echo "=== Configuring nix ==="
mkdir -p /etc/nix
printf 'build-users-group =\nsandbox = true\nexperimental-features = nix-command flakes\naccept-flake-config = true\n' > /etc/nix/nix.conf

echo "=== Installing nix ==="
mkdir -m 0755 /nix && chown root /nix
sh <(curl -L https://nixos.org/nix/install) --no-daemon
. ~/.nix-profile/etc/profile.d/nix.sh
nix --version

echo "=== Building ${VARIANT} ==="
cd /work/kernel-hub
time nix build --accept-flake-config --print-build-logs \
  ".#redistributable.${VARIANT}"

echo "=== Copying result out of ephemeral /nix ==="
mkdir -p "${OUT_DIR}"
cp -rL result/. "${OUT_DIR}/"
chmod -R u+w "${OUT_DIR}"
echo "Build complete. Artifacts:"
ls -la "${OUT_DIR}"
