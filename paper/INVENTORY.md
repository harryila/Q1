# Artifact Inventory — what supports what

For every file in this folder, this document says **what paper claim it supports**,
**which experiment it belongs to**, and **whether it is primary evidence or supporting**.

The eight key claims are numbered as in [`findings/FINDINGS.md`](findings/FINDINGS.md):

1. **Paradigm specificity of retrieval heads** (NQ vs SEC/LME)
2. **Cross-genre transfer within span-extraction paradigm** (LME → SEC)
3. **Shared retrieval substrate** (negative specificity, broad collateral damage)
4. **Semantic head clusters** (geographic/entity Jaccard cluster)
5. **Task difficulty hierarchy** (numeric vs entity ablation profiles)
6. **Priority order within shared head pools** (SEC vs LME degradation curves)
7. **Random baseline control** (ablation effects are real, not artefacts)
8. **Two independent lines of paradigm evidence** (ablation + identity)

Plus the model-replication claim (Llama → Qwen → Mistral → OLMo).

---

## Experiment 1 — Pooled ablation comparison

### Primary evidence (Llama-3.1-8B-Instruct)

| File                                                                  | Supports claims | Type |
| --------------------------------------------------------------------- | --------------- | ---- |
| `results/llama_3_1_8B_instruct/ablation/comparison_summary.json`      | 1, 7            | summary numbers (acc curves per method) |
| `results/llama_3_1_8B_instruct/ablation/accuracy_vs_knockout.png`     | 1, 7            | **Figure 1** of paper |
| `results/llama_3_1_8B_instruct/ablation/per_task_accuracy_curves.png` | 5               | **Figure 2** of paper (8 subplots) |
| `results/llama_3_1_8B_instruct/ablation/per_task_accuracy_charts/*.png` | 5             | per-task standalone plots (appendix) |
| `results/llama_3_1_8B_instruct/ablation/qrscore_sec_curves/*`         | 5, 6            | SEC-only pooled curve + heatmaps |
| `results/llama_3_1_8B_instruct/raw_results/QRScore-SEC_results.json`  | 1, 5, 6         | full per-instance accuracy curves |
| `results/llama_3_1_8B_instruct/raw_results/QRScore-8B-LME-TRAIN_results.json` | 1, 2, 6 | "                                        " |
| `results/llama_3_1_8B_instruct/raw_results/QRScore-8B-NQ-TRAIN_results.json`  | 1, 8    | "                                        " |
| `results/llama_3_1_8B_instruct/raw_results/Random-seed{42,123,456}_results.json` | 7    | random ablation control |
| `results/llama_3_1_8B_instruct/tables/accuracy_table.csv`             | 1, 5            | Method × Task × K accuracy matrix |
| `results/llama_3_1_8B_instruct/tables/drop_from_baseline_table.csv`   | 1, 5, 6         | Drop@K table |
| `results/llama_3_1_8B_instruct/tables/confidence_intervals.csv`       | 1, 7            | overall bootstrap 95% CIs (n=192, 10k iter) |
| `results/llama_3_1_8B_instruct/tables/per_task_confidence_intervals.csv` | 5            | per-task bootstrap 95% CIs |

### Replication evidence

| File                                                                 | Model               | Supports |
| -------------------------------------------------------------------- | ------------------- | -------- |
| `results/qwen_2_5_7B_instruct/ablation/{comparison_summary.json,*.png}` | Qwen-2.5-7B-Instruct | 1, 5–7 (replication) |
| `results/qwen_2_5_7B_instruct/raw_results/*_results.json`            | Qwen                | full curves, includes Transfer-* per-source |
| `results/mistral_7B_instruct/ablation/*`                             | Mistral-7B-Instruct | 1 (third-model replication) |
| `results/mistral_7B_instruct/raw_results/*`                          | Mistral             | full curves |
| `results/qwen_2_5_7B_instruct/figures_drop_vs_k/*`                   | Qwen                | per-task drop-vs-K panels (appendix) |

---

## Experiment 2 — Cross-task transfer ablation

### Primary evidence

| File                                                                   | Supports | Type |
| ---------------------------------------------------------------------- | -------- | ---- |
| `results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json` | 3, 4   | full 8×8×8 (source × target × K) drop matrix |
| `results/llama_3_1_8B_instruct/transfer/transfer_drop_heatmap_K16.png` | 3        | **Figure 3** of paper |
| `results/llama_3_1_8B_instruct/transfer/transfer_drop_heatmap_K{8,32,48,64,96,128}.png` | 3 | additional K values for appendix |
| `results/llama_3_1_8B_instruct/specificity/specificity_table.csv`      | 3        | per-source-task on/off-target drop + specificity index |
| `results/llama_3_1_8B_instruct/specificity/specificity_confidence_intervals.csv` | 3 | bootstrap CIs (4 of 6 negatives have CIs excluding 0) |
| `results/llama_3_1_8B_instruct/specificity/specificity_drop_from_k0_heatmaps_qrscore_sec.png` | 3 | **Figure 4** — multi-panel view across K |
| `results/llama_3_1_8B_instruct/specificity/specificity_raw_accuracy_heatmaps_qrscore_sec.png` | 3 | raw-accuracy version (appendix) |
| `figures/cross_ablation_curves/curve_*.png` (64 files)                 | 3, 4     | every (source, target) accuracy-vs-K curve |
| `figures/cross_ablation_curves/pair_curve_summary.csv`                 | 3, 4     | tabular summary of all 64 curves |

### Replication evidence

| File                                                       | Model               | Supports |
| ---------------------------------------------------------- | ------------------- | -------- |
| `results/qwen_2_5_7B_instruct/transfer/*`                  | Qwen                | 3 (replication) |
| `results/qwen_2_5_7B_instruct/specificity/*`               | Qwen                | 3 (replication) |
| `results/mistral_7B_instruct/transfer/*`                   | Mistral             | 3 (replication) |
| `results/mistral_7B_instruct/specificity/*`                | Mistral             | 3 (replication) |

---

## Experiment 3 — Cross-method head identity overlap

| File                                                              | Supports | Type |
| ----------------------------------------------------------------- | -------- | ---- |
| `results/llama_3_1_8B_instruct/jaccard/cross_method_head_overlap.json` | 1, 8 | SEC vs LME vs NQ Jaccard at K∈{8,16,32,48,64,96,128}, with random expected floor |
| `results/qwen_2_5_7B_instruct/jaccard/cross_method_head_overlap.json` | 1, 8  | Qwen replication |
| `results/qwen_2_5_7B_instruct/jaccard/head_similarity_jaccard.png` | 4, 8     | **Figure 6** of paper |
| `results/qwen_2_5_7B_instruct/jaccard/head_similarity_heatmaps.png` | 4       | per-K Jaccard panels |
| `results/qwen_2_5_7B_instruct/jaccard/cross_task_head_similarity_topk.json` | 4 | full 8×8 Jaccard at each K |
| `results/mistral_7B_instruct/jaccard/*`                           | 4 (replication) | |
| `results/cross_model/cross_model_head_overlap.csv`                | model-replication | per-model Jaccard pairs |
| `results/cross_model/cross_model_jaccard_pair_values.csv`         | model-replication | full pair-level values |
| `results/cross_model/cross_model_jaccard_stats.csv`               | model-replication | summary statistics |
| `results/cross_model/cross_model_jaccard_summary.pdf`             | model-replication | **Figure 7** of paper |
| `results/olmo_7B/instruct/head_similarity_jaccard.png`            | model-replication | OLMo-7B-Instruct |
| `results/olmo_7B/base/head_similarity_jaccard.png`                | model-replication | OLMo-7B (base) |

---

## Detection (head ranking) JSONs

These are the *inputs* to the ablation experiments — used to know which heads to
mask. All are small (~60 KB each).

| Folder                                          | Description |
| ----------------------------------------------- | ----------- |
| `detection/llama_3_1_8B_instruct/`              | per-task + combined SEC ranking + LME/NQ rankings + topk slices |
| `detection/qwen_2_5_7B_instruct/`               | full per-task SEC ranking + nq_train_heads + topk slices |
| `detection/mistral_7B_instruct/`                | per-task + combined + topk |
| `detection/olmo_7B/{base,instruct}/`            | per-task + combined + topk |
| `detection/external_rankings/lme_TRAIN_*.json`  | pre-computed LME rankings (Llama and Qwen) |
| `detection/external_rankings/nq_TRAIN_*.json`   | pre-computed NQ rankings (Llama and Qwen)  |

**Path convention** (per upstream README): `<model>/long_context_<task>_heads.json`
holds the full 1024-element ranking for one task; `<model>/topk/long_context_<task>_top<K>.json`
holds just the top K heads for K ∈ {8, 16, 32, 48, 64, 96, 128, 128}.

---

## Code

| Folder                                  | Purpose |
| --------------------------------------- | ------- |
| `scripts/data_prep/`                    | Build SEC train/test splits, detection instances, NIAH instances |
| `scripts/detection/detect_qrhead.py`    | Score all 1024 heads on detection data |
| `scripts/detection/run_detection.sh`    | Per-task + combined detection wrapper |
| `scripts/evaluation/run_ablation.py`    | Knockout, transfer, specificity (the paper's main pipeline) |
| `scripts/evaluation/plot_ablation.py`   | Accuracy curves + heatmaps + tables |
| `scripts/evaluation/plot_transfer.py`   | Transfer heatmaps + similarity + specificity |
| `scripts/evaluation/verify_leakage.py`  | Confirms zero filing-overlap between train/test |
| `scripts/evaluation/compute_confidence_intervals.py` | Bootstrap CI computation |
| `scripts/evaluation/compute_cross_method_overlap.py` | SEC vs LME vs NQ Jaccard |
| `scripts/evaluation/plot_task_head_jaccard.py`       | Per-task Jaccard heatmaps |
| `scripts/evaluation/plot_nq_ablation_heatmap.py`     | NQ-specific ablation heatmap |
| `src/qrretriever/`                      | Core package — `attn_retriever.py`, `custom_modeling_llama.py`, `custom_cache.py`, `config.py`, `predefined_heads.py`, `configs/` |

---

## Reference

| File | Why |
| ---- | --- |
| `reference/2506.09944v2.pdf` | Wu et al., the QRRetriever paper this submission builds on. We extend their method by (a) testing in a structurally different domain (SEC fact extraction vs. NQ/BEIR retrieval), (b) adding cross-task transfer + specificity analysis, (c) running random-baseline + bootstrap-CI controls, and (d) replicating across four open-weights models. |
| `findings/upstream_results_tex_excerpt.tex` | The `mistral_exp` branch's `results.tex` (167 lines), which contains the K-FE table (`tab:kfe`) and surrounding prose. Pulled here for reference because the published paper draft was never on `main` or the cleanup branch. |

---

## Forensics — `tab:kfe` (K-FE cross-model table)

**Question:** are the K-FE numbers in `tab:kfe` (Llama–Qwen 0.18 / 0.02,
Llama–Mistral 0.47 / 0.04, Qwen–Mistral 0.59 / 0.15) actually computed from
data in this repo, or did the analysis happen off-repo?

**Answer:** the numbers are **reproducible from data we already have**, but
the **original code path was not preserved**. The script that lives in our
repo (`scripts/evaluation/compute_kfe_correlations.py`) is a *forensic
reconstruction* committed to `mistral_exp` after the fact, not the original
production code.

| File                                                              | What it is |
| ----------------------------------------------------------------- | ---------- |
| `scripts/evaluation/compute_target_sensitivity.py`                | Builds the (task × K) sensitivity / efficacy matrices from each model's `cross_task_transfer_matrix.json`. Source-of-truth for the matrix definitions. |
| `scripts/evaluation/compute_kfe_correlations.py`                  | Computes the K-FE R² values reported in `tab:kfe`. The script's docstring explicitly states: *"Because the original definition of 'efficacy' wasn't preserved in code anywhere in the repo, we evaluate four plausible source-centric measures and report all of them."* It tests four candidate efficacy definitions and reports which best matches the published values. |
| `results/cross_model/kfe/kfe_table.csv`                           | Long-form output: derived K-FE R² for sensitivity (one definition) and efficacy (four definitions) vs the published values, plus absolute deviations. |
| `results/cross_model/kfe/kfe_report.md`                           | Human-readable verdict. |

**What was confirmed:**

- **Sensitivity (target-centric)** — total |Δ| from published = **0.008**.
  Definition is unambiguous: `sensitivity[t][K] = mean over sources s of drop[s][t][K]`.
  Reproduces 0.18 / 0.47 / 0.59 to 3 decimals.
- **Efficacy (source-centric)** — only one of four candidate definitions
  reproduces the published numbers: `efficacy[s][K] = mean over t≠s of drop[s][t][K]`
  (i.e. `off_target_mean`). Total |Δ| = **0.007** with that definition.
  The other three candidates miss by 0.13–0.82.

**What this means for the paper:**

- The published `tab:kfe` numbers can be regenerated from
  `cross_task_transfer_matrix.json` files we already ship for every model.
- The residualization step is straightforward: subtract within-K column
  means from each (task, K) cell, flatten to a 56-element vector, take
  Pearson R² between two models. No hidden model-fitting or off-repo data.
- **Caveat to flag in writing:** the "efficacy" definition the paper uses
  is *off-target mean drop*, but the table caption only says "source
  efficacy" without a formula. We should either (a) inline the formula in
  the caption, or (b) cite `compute_kfe_correlations.py` for the full
  definition. The current draft does neither.

**To re-run (one command):**

```bash
python scripts/evaluation/compute_kfe_correlations.py \
  --inputs \
    "Llama-3.1-8B-Instruct=results/llama_3_1_8B_instruct/transfer/cross_task_transfer_matrix.json" \
    "Qwen2.5-7B-Instruct=results/qwen_2_5_7B_instruct/transfer/cross_task_transfer_matrix.json" \
    "Mistral-7B-Instruct-v0.3=results/mistral_7B_instruct/transfer/cross_task_transfer_matrix.json" \
  --output_dir results/cross_model/kfe
```

---

## Quick numerical sanity check (data → claim)

If you want to verify numbers as you write, here are the source-of-truth files:

| Number in paper                                              | Read from |
| ------------------------------------------------------------ | --------- |
| Llama K=0 baseline = 91.1% on n=192                          | `results/llama_3_1_8B_instruct/ablation/comparison_summary.json` → `methods.QRScore-SEC.baseline_accuracy` |
| Llama QRScore-SEC drop@K=16 = 51.5pp                         | same JSON, `methods.QRScore-SEC.drop_at_k16` |
| Llama random baseline curve                                  | same JSON, `methods.Random-seed{42,123,456}.accuracy_curve` |
| Bootstrap 95% CIs for any (method, K)                        | `results/llama_3_1_8B_instruct/tables/confidence_intervals.csv` |
| Per-task K=8 / K=16 / K=32 numbers                           | `results/llama_3_1_8B_instruct/tables/accuracy_table.csv` |
| Specificity index per source task with CI                    | `results/llama_3_1_8B_instruct/specificity/specificity_table.csv` and `specificity_confidence_intervals.csv` |
| Jaccard top-K SEC vs LME vs NQ                               | `results/llama_3_1_8B_instruct/jaccard/cross_method_head_overlap.json` |
| Per-task 8×8 Jaccard at each K                               | `results/qwen_2_5_7B_instruct/jaccard/cross_task_head_similarity_topk.json` (also in mistral & OLMo dirs) |
| Cross-task transfer accuracy(source, target, K)              | `results/<model>/transfer/cross_task_transfer_matrix.json` |
| Qwen K=0 baseline = 89.2% on n=241                           | `results/qwen_2_5_7B_instruct/ablation/comparison_summary.json` |
| Cross-model summary plot                                     | `results/cross_model/cross_model_jaccard_summary.pdf` |

---

## Pre-submission analyses (added after the original submission scaffolding)

Five analyses added to pre-empt the strongest reviewer objections. Each row maps an objection to the artifact that resolves it. The plan that produced these lives at `~/.cursor/plans/five_pre-submission_analyses_*.plan.md` (outside the repo).

| # | Reviewer objection it pre-empts | Code | Output | Verdict (this run) |
| ---: | --- | --- | --- | --- |
| **2** | "Fragility is just baseline difficulty in disguise." | [`scripts/analysis/predictive_fragility.py`](scripts/analysis/predictive_fragility.py) | [`results/fragility_predictors/{correlations.csv, scatter_plots.png, REPORT.md}`](results/fragility_predictors/) | **SHARP** — \|r_baseline\|=0.093, p=0.827. Fragility does **not** correlate with baseline accuracy. Query→answer distance does correlate strongly negatively (r=−0.720, p=0.044) — flag in §5 limitations. |
| **5** | "tab:kfe numbers can't be reproduced from the repo." | [`scripts/evaluation/{verify_kfe.sh, verify_kfe.py, compute_kfe_correlations.py, compute_target_sensitivity.py}`](scripts/evaluation/) | [`results/cross_model/kfe/{kfe_table.csv, kfe_report.md, VERIFICATION.md}`](results/cross_model/kfe/) | **PASS** — all 6 asserted rows reproduce within EPS=0.05 (max Δ=0.0035). |
| **4** | "Cross-model retrieval-head pools — varies vs random alignment, or vs expected alignment?" | [`scripts/analysis/cross_model_union_overlap.py`](scripts/analysis/cross_model_union_overlap.py) | [`results/cross_model/{cross_model_union_overlap.csv, cross_model_union_overlap_plot.png, cross_model_union_overlap_REPORT.md}`](results/cross_model/) | **WEAKER THAN HYPOTHESISED** — at K=16, lifts are 0.0×–1.7× across pairs (not the 5–10× we hypothesised). §4 prose pivots to **per-model phenomenon**, not a shared substrate claim. |
| **1** | "Tasks differ in fragility — but is the fragility signature *layer-localised*?" | [`scripts/analysis/layer_distribution.py`](scripts/analysis/layer_distribution.py) | [`results/layer_analysis/{REPORT.md, per_task_layer_histograms.png, fragile_vs_robust_overlay.png, layer_concentration_table.csv, permutation_tests.csv}`](results/layer_analysis/) | **NOT SIGNIFICANT** — best p-value across 5 models is 0.107 (OLMo-base); permutation test does not detect a layer-localised circuit. §4 prose reports histograms descriptively without a mechanistic claim. |
| **3** | "Any partially-overlapping detection method would look like this — QRScore might just be picking 16 heads with non-zero relevance by chance." | [`scripts/evaluation/{random_head_null.py, random_head_null.sh}`](scripts/evaluation/) | [`results/random_head_null/REQUIRES_GPU.md`](results/random_head_null/REQUIRES_GPU.md) (script + wrapper + run instructions) | **GPU PENDING** — script and wrapper are committed, runs require CUDA + HF cache. See [`REQUIRES_GPU.md`](results/random_head_null/REQUIRES_GPU.md) for the four-tier deployment table keyed off smoke-test timing. |

### What this means for the paper draft

- **Tasks 5, 2 land favourably**: K-FE table reproduces, baseline-difficulty objection killed.
- **Tasks 1, 4 land in the fallback case**: no significant layer localisation; no shared cross-model substrate at K=16. Both findings reframe §4 from declarative to descriptive — defensible per the CFP, but the paper draft needs to be written for the **fallback narratives**, not the favourable ones.
- **Task 3 outstanding**: GPU-only; script ready to run. Pooled SEC ablation claim (already supported by existing data) is unaffected by the outcome.

### Pre-submission analysis source files

| Path | Purpose |
| ---- | ------- |
| `scripts/analysis/predictive_fragility.py` | Task 2 — fragility vs baseline / query-distance / answer-length |
| `scripts/analysis/cross_model_union_overlap.py` | Task 4 — Jaccard + overlap-coefficient on per-model union of top-K heads |
| `scripts/analysis/layer_distribution.py` | Task 1 — layer entropy + permutation + Mann-Whitney U |
| `scripts/evaluation/verify_kfe.sh` | Task 5 — wrapper for `compute_kfe_correlations.py` against local matrices |
| `scripts/evaluation/verify_kfe.py` | Task 5 — assertion harness; nonzero exit if `tab:kfe` drifts past EPS=0.05 |
| `scripts/evaluation/random_head_null.py` | Task 3 — random-16-head ablation null distribution (GPU) |
| `scripts/evaluation/random_head_null.sh` | Task 3 — four-tier wrapper: smoke / full / half / llama |
| `data/niah_input/*_test.json` | Task 2 input — re-fetched 8 NIAH evaluation files (~15 MB) |
