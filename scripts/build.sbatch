#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --job-name=ta-kernel-build
#SBATCH --output=ta-kernel-build-%j.out

# Variant to build (override via: sbatch --export=VARIANT=... build.sbatch)
VARIANT="${VARIANT:-torch211-cxx11-cu128-x86_64-linux}"

# Ensure persistent nix store dir exists on host so /nix survives container teardown
mkdir -p "${HOME}/nix-store-cache"

srun --container-image='docker://debian:bookworm-slim' \
     --container-mounts="${HOME}/ThriftAttention:/work,${HOME}/nix-store-cache:/nix" \
     --container-workdir=/work \
     --container-remap-root \
     --container-writable \
     --export=ALL,VARIANT="${VARIANT}" \
     bash /work/scripts/build-in-container.sh
