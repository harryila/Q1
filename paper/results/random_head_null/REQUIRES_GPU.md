# Random-head null distribution — GPU required

This directory will hold the outputs of Task 3 (random source-head null
distribution). The analysis script
[`scripts/evaluation/random_head_null.py`](../../scripts/evaluation/random_head_null.py)
and wrapper
[`scripts/evaluation/random_head_null.sh`](../../scripts/evaluation/random_head_null.sh)
are written and committed; **the runs themselves require a CUDA GPU and a
HuggingFace cache containing Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct,
and Mistral-7B-Instruct-v0.3.**

## What this analysis answers

Reviewer objection it pre-empts: *"Any partially-overlapping detection
method would look like this — your QRScore detection might just be picking
16 heads with non-zero relevance by chance."*

Concretely: for each model and each of N random 16-head subsets, run the
same head-masking + generation pipeline as `run_ablation.py`, and measure
per-task efficacy (mean off-target drop). The resulting per-task null
distribution is compared against the observed per-task efficacy from
`results/<model>/transfer/cross_task_transfer_matrix.json`.

## How to run (on a GPU box)

```bash
# 0. Clone / pull this repo on the GPU box, then cd into paper/
git clone https://github.com/harryila/Q1.git
cd Q1/paper
pip install -e src/                       # pulls torch, transformers, flash_attn
pip install numpy scipy matplotlib

# 1. Smoke first (gates the full-run scope)
bash scripts/evaluation/random_head_null.sh smoke
#   - 3 random subsets on Llama-3.1-8B-Instruct
#   - prints per-instance generation time
#   - takes ~5-15 minutes total

# 2. Choose the deployment tier from the smoke timing:
#       <= 1 sec/instance  -> bash scripts/evaluation/random_head_null.sh full     (100 x 3 models)
#       1-2 sec/instance   -> bash scripts/evaluation/random_head_null.sh full     (run over a weekend)
#       2-4 sec/instance   -> bash scripts/evaluation/random_head_null.sh half     (50 x 3 models)
#       > 4 sec/instance   -> bash scripts/evaluation/random_head_null.sh llama    (Llama-only at 50)

# 3. Push results back
git add results/random_head_null/
git commit -m "Task 3: random-head null distribution outputs (tier: <chosen>)"
git push
```

## Why we didn't run it locally

The CPU-only host this repo is being prepared on cannot load Llama-7B at
useful speed. The smoke run above will measure per-instance time on the
target GPU; the four-tier table in
[`scripts/evaluation/random_head_null.py`](../../scripts/evaluation/random_head_null.py)
docstring will then determine which tier fits the available GPU envelope.

## Compute envelope (math, not wall-clock)

`100 random subsets × 3 models × 8 tasks × 24 instances × T sec/instance`
= `57600 × T sec` total.

| T (sec/instance) | Total wall-clock |
| ---: | --- |
| 0.5 | ~8 GPU-hours |
| 1.0 | ~16 GPU-hours |
| 2.0 | ~32 GPU-hours |
| 5.0 | ~80 GPU-hours |

The previous `15 GPU-hours` estimate in older drafts of the plan was an
arithmetic error — corrected here.

## Expected outputs (per model)

After a successful run the directory `results/random_head_null/<slug>/`
will contain:

- `null_samples.json` — per-sample dict `{seed, heads, per_task_acc, per_task_drop, per_task_efficacy, elapsed_seconds}`.
- `null_summary.csv` — per task: mean / p2.5 / p50 / p97.5 of drop and efficacy.
- `observed_vs_null.csv` — per task: observed efficacy from the local transfer matrix vs the null mean / p95 / observed-percentile / above-p95 flag.
- `per_task_null_vs_observed.png` — violin of null with observed marker per task.
- `REPORT.md` — verdict + provenance.

## Verdict logic

- If `>=7/8` tasks across `>=2/3` models have observed efficacy above the 95th percentile of their null, the headline claim is: *"Observed task-specific efficacy lies above the 95th percentile of the random-16-head null in N/8 tasks per model — QRScore detection is provably non-random."*
- If the count is mixed, per-task efficacy is real for some tasks and at chance for others. **The pooled SEC ablation claim (already in `findings/FINDINGS.md` Experiment 1) is unaffected** because the QRScore-SEC drop curve is already separated from random by ~50pp at K=16 in the existing data.
