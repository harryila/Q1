# Layer-position analysis — REPORT

## Setup

- Top-16 heads per (model, task), parsed as `(layer, head)` from `detection/<model>/topk/long_context_<task>_top16.json` (or sliced from the full ranking when the top-K file isn't present, e.g. for 3 Llama tasks).
- Models analysed: Llama-3.1-8B-Instruct, Mistral-7B-Instruct, OLMo-7B-Instruct, OLMo-7B (base), Qwen2.5-7B-Instruct.
- Unit of analysis: per-task layer entropy (one number per task → n=4 fragile vs n=4 robust). Pooling 4 × 16 = 64 head-layer indices for a KS test would violate independence (tasks share heads — `headquarters_city` ↔ `headquarters_state` overlap at 78% per FINDINGS.md §3).
- Fragility ranking from Task 2 (`results/fragility_predictors/correlations.csv`):

| Rank | Task | Fragility |
| ---: | --- | ---: |
| 1 | `employees_count_total`  ← fragile | 0.692 |
| 2 | `incorporation_year`  ← fragile | 0.473 |
| 3 | `ceo_lastname`  ← fragile | 0.431 |
| 4 | `incorporation_state`  ← fragile | 0.415 |
| 5 | `holder_record_amount`  ← robust | 0.398 |
| 6 | `headquarters_city`  ← robust | 0.336 |
| 7 | `headquarters_state`  ← robust | 0.218 |
| 8 | `registrant_name`  ← robust | 0.110 |

## Per-model verdict (permutation + Mann-Whitney U on per-task entropies)

| Model | Mean fragile entropy | Mean robust entropy | Δ obs | Perm p | MW-U p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 1.561 | 1.842 | -0.282 | 0.293 | 0.343 |
| Mistral-7B-Instruct | 1.900 | 1.882 | +0.018 | 0.84 | 0.885 |
| OLMo-7B-Instruct | 1.798 | 1.859 | -0.061 | 0.48 | 0.663 |
| OLMo-7B (base) | 1.526 | 1.399 | +0.126 | 0.107 | 0.108 |
| Qwen2.5-7B-Instruct | 1.738 | 1.718 | +0.021 | 0.942 | 0.686 |

## Verdict logic

- If both permutation p < 0.05 AND Mann-Whitney U p < 0.05 in 3+ of the 5 models, **claim**: "Per-task layer entropy differs significantly between fragile and robust tasks (permutation p<0.05 in N/5 models, Mann-Whitney consistent direction). The fragility signature is layer-localised."
- If only the permutation test rejects (likely given Mann-Whitney's n=4 floor), **claim**: "Permutation test detects systematic difference; Mann-Whitney is underpowered with n=4 vs n=4."
- If neither rejects consistently, **fallback**: drop the layer-localisation claim; report the histograms descriptively in §4 without a mechanistic narrative.

## Cross-model layer-mass agreement (per task, 14-bin relative-depth Pearson r)

| Task | Llama↔Mistral | Llama↔OLMo-Inst | Llama↔Qwen | Mistral↔OLMo-Inst | Mistral↔Qwen | OLMo-Inst↔Qwen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ceo_lastname` | +0.01 | -0.38 | -0.08 | +0.18 | -0.18 | +0.44 |
| `employees_count_total` | +0.42 | -0.23 | -0.36 | +0.15 | -0.01 | +0.44 |
| `headquarters_city` | +0.55 | +0.04 | +0.09 | -0.01 | +0.29 | +0.41 |
| `headquarters_state` | +0.44 | -0.19 | -0.26 | +0.13 | +0.31 | +0.37 |
| `holder_record_amount` | +0.36 | -0.16 | -0.14 | +0.15 | +0.23 | +0.81 |
| `incorporation_state` | +0.20 | -0.32 | -0.29 | +0.01 | +0.54 | +0.46 |
| `incorporation_year` | +0.29 | -0.32 | -0.32 | +0.01 | +0.46 | +0.52 |
| `registrant_name` | +0.05 | -0.19 | -0.34 | -0.06 | +0.53 | +0.36 |

## Files

- `per_task_layer_histograms.png`: 5×8 grid (rows=models, cols=tasks).
- `fragile_vs_robust_overlay.png`: per-model fragile-vs-robust bars on the relative-depth axis with permutation/MW-U p-values annotated.
- `layer_concentration_table.csv`: per (model, task) entropy / mode / IQR.
- `permutation_tests.csv`: per model, observed Δ entropy + permutation null mean/std/p-value + Mann-Whitney U statistic and p-value + cross-model Pearson r per task.
