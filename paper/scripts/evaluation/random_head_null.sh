#!/usr/bin/env bash
# Wrapper for random_head_null.py. Loops over the three local-result models
# and writes per-model outputs into results/random_head_null/<model_slug>/.
#
# GPU REQUIRED. Run the smoke step first, then choose deployment tier from
# the timing it reports. See REQUIRES_GPU.md.
#
# Usage:
#     bash scripts/evaluation/random_head_null.sh smoke
#     bash scripts/evaluation/random_head_null.sh full     # 100 random subsets x 3 models, K=16
#     bash scripts/evaluation/random_head_null.sh half     # 50 x 3 models, K=16
#     bash scripts/evaluation/random_head_null.sh llama    # 50 random subsets x Llama only
#
# Output tree:
#   results/random_head_null/_smoke/                                (smoke)
#   results/random_head_null/<model_slug>/                          (full / half / llama)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-smoke}"

# (model_id, local_result_dir_slug)
LLAMA="meta-llama/Llama-3.1-8B-Instruct llama_3_1_8B_instruct"
QWEN="Qwen/Qwen2.5-7B-Instruct          qwen_2_5_7B_instruct"
MISTRAL="mistralai/Mistral-7B-Instruct-v0.3 mistral_7B_instruct"

run_one() {
    local model_id="$1"
    local slug="$2"
    local n="$3"
    local out_dir="results/random_head_null/${slug}"
    mkdir -p "$out_dir"
    python scripts/evaluation/random_head_null.py \
        --model "$model_id" \
        --niah_dir data/niah_input \
        --transfer_matrix "results/${slug}/transfer/cross_task_transfer_matrix.json" \
        --baseline_accuracy_json "results/${slug}/raw_results/QRScore-SEC_results.json" \
        --num_random_subsets "$n" \
        --K 16 \
        --max_instances_per_task 24 \
        --max_context_tokens 8192 \
        --output_dir "$out_dir" \
        --seed_base 1000
}

case "$MODE" in
    smoke)
        echo "=== Smoke: 3 random subsets on Llama only ==="
        run_one $LLAMA 3
        echo
        echo "Smoke done. Read scripts/evaluation/random_head_null.py output (samples[*].elapsed_seconds)"
        echo "to choose the deployment tier:"
        echo "    <= 1 sec/instance  -> bash $0 full"
        echo "    1-2 sec/instance   -> bash $0 full     (run over a weekend)"
        echo "    2-4 sec/instance   -> bash $0 half"
        echo "    > 4 sec/instance   -> bash $0 llama"
        ;;
    full)
        echo "=== Full: 100 x 3 models ==="
        run_one $LLAMA   100
        run_one $QWEN    100
        run_one $MISTRAL 100
        ;;
    half)
        echo "=== Half: 50 x 3 models ==="
        run_one $LLAMA   50
        run_one $QWEN    50
        run_one $MISTRAL 50
        ;;
    llama)
        echo "=== Llama-only: 50 random subsets ==="
        run_one $LLAMA   50
        ;;
    *)
        echo "Usage: $0 [smoke|full|half|llama]" >&2
        exit 1
        ;;
esac

echo
echo "Done. Outputs are in results/random_head_null/<slug>/"
