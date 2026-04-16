#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────
#  LSFB-ISOL automated pipeline (Docker init container)
# ─────────────────────────────────────────────────────────
#  1. Download  LSFB-ISOL poses from lsfb.info.unamur.be
#  2. Convert   to SignFlow 225-dim landmark format
#  3. Seed      signs + videos into PostgreSQL
#  4. Train     SignTransformer fine-tuning on LSFB-ISOL
#  5. Exit 0 so the backend service can start
# ─────────────────────────────────────────────────────────
set -euo pipefail

export PYTHONPATH="/app:${PYTHONPATH:-}"

# Scripts directory: mounted from repo-root scripts/ or fallback to /app/scripts
SCRIPTS_DIR="/app/scripts/repo"
if [ ! -d "$SCRIPTS_DIR" ]; then
    SCRIPTS_DIR="/app/scripts"
fi

LSFB_DIR="${LSFB_DATA_DIR:-/app/data/datasets/lsfb_isol}"
MODEL_DIR="${MODEL_DIR:-/app/data/models}"
CONVERTED_DIR="${LSFB_DIR}/converted"
MODEL_OUTPUT="${MODEL_DIR}/lsfb_isol_finetuned.pt"
MARKER="${LSFB_DIR}/.pipeline_done"

# Training hyperparameters (overridable via env)
LSFB_MAX_SIGNS="${LSFB_MAX_SIGNS:-}"
LSFB_MIN_CLIPS="${LSFB_MIN_CLIPS:-10}"
LSFB_EPOCHS="${LSFB_EPOCHS:-80}"
LSFB_BATCH_SIZE="${LSFB_BATCH_SIZE:-64}"
LSFB_LR="${LSFB_LR:-1e-4}"
LSFB_DEVICE="${LSFB_DEVICE:-cpu}"
LSFB_CHECKPOINT="${LSFB_CHECKPOINT:-}"
LSFB_FREEZE_LAYERS="${LSFB_FREEZE_LAYERS:-2}"
LSFB_AUGMENTATIONS="${LSFB_AUGMENTATIONS:-4}"

# Skip pipeline if already done (persistent volume)
if [ -f "$MARKER" ] && [ -f "$MODEL_OUTPUT" ]; then
    echo "══════════════════════════════════════════════════════════"
    echo " LSFB pipeline already completed — skipping."
    echo " Remove ${MARKER} to force re-run."
    echo "══════════════════════════════════════════════════════════"
    exit 0
fi

echo "══════════════════════════════════════════════════════════"
echo "  LSFB-ISOL Auto-Pipeline"
echo "══════════════════════════════════════════════════════════"
echo "  Data dir:     ${LSFB_DIR}"
echo "  Model output: ${MODEL_OUTPUT}"
echo "  Device:       ${LSFB_DEVICE}"
echo "  Epochs:       ${LSFB_EPOCHS}"
echo "  Max signs:    ${LSFB_MAX_SIGNS:-all}"
echo "══════════════════════════════════════════════════════════"

# ── Helpers ──────────────────────────────────────────────

wait_for_postgres() {
    echo "[init] Waiting for PostgreSQL..."
    local retries=30
    while [ $retries -gt 0 ]; do
        python -c "
from sqlalchemy import create_engine, text
import os
e = create_engine(os.environ['DATABASE_URL'])
with e.connect() as c:
    c.execute(text('SELECT 1'))
" 2>/dev/null && break
        retries=$((retries - 1))
        sleep 2
    done
    if [ $retries -eq 0 ]; then
        echo "[init] ERROR: PostgreSQL not available after 60s" >&2
        exit 1
    fi
    echo "[init] PostgreSQL is ready."
}

# ── Step 1: Download ─────────────────────────────────────

step_download() {
    local instances_csv="${LSFB_DIR}/instances.csv"
    if [ -f "$instances_csv" ]; then
        local file_count
        file_count=$(find "${LSFB_DIR}/poses" -name "*.npy" 2>/dev/null | wc -l || echo 0)
        if [ "$file_count" -gt 100000 ]; then
            echo "[1/4] Dataset already downloaded (${file_count} pose files) — skipping."
            return 0
        fi
        echo "[1/4] Partial download detected (${file_count} files). Resuming..."
    else
        echo "[1/4] Downloading LSFB-ISOL poses..."
    fi

    PYTHONWARNINGS="ignore:Unverified HTTPS request" \
        python "${SCRIPTS_DIR}/download_lsfb.py" \
        --destination "${LSFB_DIR}" \
        --no-ssl-verify \
        --skip-existing
}

# ── Step 2: Convert ──────────────────────────────────────

step_convert() {
    local report="${CONVERTED_DIR}/conversion_report.json"
    if [ -f "$report" ]; then
        local converted_count
        converted_count=$(python -c "import json; print(json.load(open('${report}'))['success'])" 2>/dev/null || echo 0)
        if [ "$converted_count" -gt 0 ]; then
            echo "[2/4] Conversion already done (${converted_count} instances). Re-converting new files..."
        fi
    fi

    echo "[2/4] Converting LSFB-ISOL poses to SignFlow format..."
    python "${SCRIPTS_DIR}/convert_lsfb.py" \
        --lsfb-dir "${LSFB_DIR}" \
        --output-dir "${CONVERTED_DIR}"
}

# ── Step 3: Seed DB ──────────────────────────────────────

step_seed() {
    echo "[3/4] Seeding database with LSFB-ISOL signs and videos..."
    wait_for_postgres

    python "${SCRIPTS_DIR}/seed_lsfb.py" \
        --lsfb-dir "${LSFB_DIR}" \
        --converted-dir "${CONVERTED_DIR}"
}

# ── Step 4: Train ────────────────────────────────────────

step_train() {
    if [ -f "$MODEL_OUTPUT" ]; then
        echo "[4/4] Fine-tuned model already exists at ${MODEL_OUTPUT}."
        echo "      Delete it to force re-training."
        return 0
    fi

    echo "[4/4] Fine-tuning SignTransformer on LSFB-ISOL..."

    local train_args=(
        --lsfb-dir "${LSFB_DIR}"
        --converted-dir "${CONVERTED_DIR}"
        --epochs "${LSFB_EPOCHS}"
        --batch-size "${LSFB_BATCH_SIZE}"
        --lr "${LSFB_LR}"
        --device "${LSFB_DEVICE}"
        --output "${MODEL_OUTPUT}"
        --min-clips-per-sign "${LSFB_MIN_CLIPS}"
        --freeze-layers "${LSFB_FREEZE_LAYERS}"
        --augmentations "${LSFB_AUGMENTATIONS}"
    )

    if [ -n "${LSFB_MAX_SIGNS}" ]; then
        train_args+=(--max-signs "${LSFB_MAX_SIGNS}")
    fi

    if [ -n "${LSFB_CHECKPOINT}" ] && [ -f "${LSFB_CHECKPOINT}" ]; then
        train_args+=(--checkpoint "${LSFB_CHECKPOINT}")
    fi

    python "${SCRIPTS_DIR}/finetune_lsfb.py" "${train_args[@]}"
}

# ── Run pipeline ─────────────────────────────────────────

mkdir -p "${LSFB_DIR}" "${MODEL_DIR}"

step_download
step_convert
step_seed
step_train

# Mark pipeline as complete
date -Iseconds > "$MARKER"

echo ""
echo "══════════════════════════════════════════════════════════"
echo "  LSFB-ISOL pipeline completed successfully!"
echo "  Model: ${MODEL_OUTPUT}"
echo "══════════════════════════════════════════════════════════"
