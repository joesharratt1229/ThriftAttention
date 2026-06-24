#!/bin/bash
set -euo pipefail

VARIANT="${VARIANT:-torch211-cxx11-cu128-x86_64-linux}"
OUT_DIR="/work/build-output/${VARIANT}"

echo "=== Installing apt deps ==="
apt update && apt install -y git curl xz-utils ca-certificates

echo "=== Configuring nix ==="
mkdir -p /etc/nix
printf 'build-users-group =\nsandbox = true\nexperimental-features = nix-command flakes\naccept-flake-config = true\nmax-jobs = auto\ncores = 0\n' > /etc/nix/nix.conf

if [ -x /nix/var/nix/profiles/default/bin/nix ]; then
  echo "=== Reusing existing nix install in /nix ==="
  export PATH=/nix/var/nix/profiles/default/bin:${PATH}
else
  echo "=== Installing nix into persistent /nix ==="
  # /nix exists (bind-mounted) but may be empty; chown so the installer can write
  chown root:root /nix
  sh <(curl -L https://nixos.org/nix/install) --no-daemon
  . ~/.nix-profile/etc/profile.d/nix.sh
fi

nix --version

echo "=== Building ${VARIANT} ==="
cd /work
time nix build --accept-flake-config --print-build-logs \
  ".#redistributable.${VARIANT}"

echo "=== Copying result out of ephemeral /nix ==="
mkdir -p "${OUT_DIR}"
cp -rL result/. "${OUT_DIR}/"
chmod -R u+w "${OUT_DIR}"
echo "Build complete. Artifacts:"
ls -la "${OUT_DIR}"
