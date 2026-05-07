#!/usr/bin/env bash
# Re-run the K-FE residualization on local matrices and compare against the
# published values in tab:kfe (results.tex). Used by reviewers to confirm
# reproducibility from the repo with one command.
#
# Usage:
#     bash scripts/evaluation/verify_kfe.sh
#
# Exits 0 if every derived value lands within the assertion harness's EPS
# tolerance of the published value; nonzero otherwise (with a diff table).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="results/cross_model/kfe"
mkdir -p "$OUTPUT_DIR"

echo "=== Running compute_kfe_correlations.py against local matrices ==="
python scripts/evaluation/compute_kfe_correlations.py \
    --inputs \
        "Llama-3.1-8B-Instruct=results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json" \
        "Qwen2.5-7B-Instruct=results/qwen_2_5_7B_instruct/transfer/cross_task_transfer_matrix.json" \
        "Mistral-7B-Instruct-v0.3=results/mistral_7B_instruct/transfer/cross_task_transfer_matrix.json" \
    --output_dir "$OUTPUT_DIR"

echo
echo "=== Asserting derived values match published tab:kfe (EPS=0.05) ==="
python scripts/evaluation/verify_kfe.py --table "$OUTPUT_DIR/kfe_table.csv"

echo
echo "K-FE verification PASSED. Outputs:"
echo "  $OUTPUT_DIR/kfe_table.csv"
echo "  $OUTPUT_DIR/kfe_report.md"
