#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-0.001}"
DATA_ROOT="${DATA_ROOT:-$SCRIPT_DIR/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/outputs}"
DEVICE="${DEVICE:-mps}"
N_MELS_VALUES="${N_MELS_VALUES:-20,40,80}"
GROUPS_VALUES="${GROUPS_VALUES:-2,4,8,16}"

python "$SCRIPT_DIR/train_pipeline.py" \
  --mode all \
  --wav-path "$SCRIPT_DIR/file_for_test.wav" \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --n-mels-values "$N_MELS_VALUES" \
  --groups-values "$GROUPS_VALUES" \
  --baseline-n-mels 80

cat <<EOF
Run completed.
Output directory: $OUTPUT_DIR
- logmel_reference_compare.png
- n_mels_results.csv
- n_mels_train_loss.png
- n_mels_test_accuracy.png
- groups_results.csv
- groups_metrics.png
EOF
