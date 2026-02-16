#!/usr/bin/env bash

set -euo pipefail

if ! command -v torch-model-archiver >/dev/null 2>&1; then
  echo "torch-model-archiver is required. Install torchserve tooling first."
  exit 1
fi

MODEL_NAME="${MODEL_NAME:-signflow}"
MODEL_VERSION="${MODEL_VERSION:-1.0}"
MODEL_PATH="${1:-/app/data/models/model_v1.pt}"
HANDLER_PATH="${HANDLER_PATH:-/app/torchserve/handler.py}"
LABELS_PATH="${LABELS_PATH:-/app/torchserve/labels.json}"
EXPORT_PATH="${EXPORT_PATH:-/app/torchserve/model-store}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Model file not found: ${MODEL_PATH}"
  exit 1
fi

mkdir -p "${EXPORT_PATH}"

torch-model-archiver \
  --model-name "${MODEL_NAME}" \
  --version "${MODEL_VERSION}" \
  --serialized-file "${MODEL_PATH}" \
  --handler "${HANDLER_PATH}" \
  --extra-files "${LABELS_PATH}" \
  --export-path "${EXPORT_PATH}" \
  --force

echo "Model archive created in ${EXPORT_PATH}/${MODEL_NAME}.mar"
