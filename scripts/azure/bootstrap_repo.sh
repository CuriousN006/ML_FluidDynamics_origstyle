#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$HOME/ML_FluidDynamics_azure_gpu}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "python3.12 is required on the VM. Re-run provisioning or install it first." >&2
  exit 1
fi

if [[ ! -d "$REPO_DIR" ]]; then
  echo "Repo directory not found: $REPO_DIR" >&2
  exit 1
fi

cd "$REPO_DIR"

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install --index-url "$TORCH_INDEX_URL" torch
python -m pip install -e .

if [[ -f CYLINDER_ALL.mat ]]; then
  python -m mlfd.setup
else
  echo "CYLINDER_ALL.mat is not present in $REPO_DIR yet." >&2
  echo "Upload the dataset with scripts/azure/upload_data.ps1, then rerun validation." >&2
fi
