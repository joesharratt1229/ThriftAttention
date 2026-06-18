#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ta-kernel-build
#SBATCH --output=ta-kernel-build-%j.out

# Variant to build (override via: sbatch --export=VARIANT=... build.sbatch)
VARIANT="${VARIANT:-torch212-cxx11-cu130-x86_64-linux}"

srun --container-image='docker://debian:bookworm-slim' \
     --container-mounts="${HOME}/ThriftAttention:/work" \
     --container-workdir=/work/kernel-hub \
     --container-remap-root \
     --container-writable \
     --export=ALL,VARIANT="${VARIANT}" \
     bash /work/kernel-hub/scripts/build-in-container.sh
