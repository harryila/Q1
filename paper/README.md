# Paper assets — ICML 2026 Mechanistic Interpretability Workshop

Workshop: <https://mechinterpworkshop.com/> · CFP: <https://mechinterpworkshop.com/cfp/>
LaTeX template: `../icml2026/` · Submission deadline: **May 8, 2026 (AOE)**

This folder is a curated, paper-ready snapshot of the experimental artifacts from
[`4n4ny4/qr_scoring`](https://github.com/4n4ny4/qr_scoring) (across all branches).
The original repo is sprawling — multiple branches, multi-GB raw CSVs, megabyte-sized
token logs, `.DS_Store`s, and duplicated trees. Here we keep only what is needed to
write, support, and reproduce the workshop submission.

---

## TL;DR — One-paragraph paper pitch

We test the QRRetriever framework's claim that a small set of attention heads
("retrieval heads") drives long-context retrieval, by **detecting** these heads
on SEC 10-K filings and then **knocking them out** to measure the accuracy drop.
Across **four open-weights models** — Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct,
Mistral-7B-Instruct, and OLMo-7B (base + instruct) — and three head-ranking sources
(SEC, LME, NQ), we find:

1. **Head importance is paradigm-specific, not domain-specific.** Heads detected
   on **passage-sorting** tasks (NQ/BEIR) barely transfer to **span-extraction**
   tasks (SEC, LME). Random ablation is statistically indistinguishable from NQ
   ablation at K=16.
2. **Heads form a shared retrieval substrate, not task-specific circuits.**
   Specificity indices are negative for 6/8 SEC tasks (4 with bootstrap CIs that
   exclude zero) — knocking out one task's "top heads" damages other tasks as much
   or more.
3. **Functional clusters exist for semantically related tasks.** Geographic/entity
   tasks (`headquarters_city`, `headquarters_state`, `registrant_name`) share
   45–78% of top-16 heads; numeric tasks share <7%.
4. **The paradigm-specificity finding replicates across model families** (Llama,
   Qwen, Mistral, OLMo) — see `results/cross_model/`.

Headline numbers (Llama-3.1-8B-Instruct, n=192, K=16):

| Method                       | K=0 acc | K=16 acc          | 95% CI         |
| ---------------------------- | ------- | ----------------- | -------------- |
| QRScore-SEC (in-domain)      | 91.1%   | **39.6%** (Δ51.5) | [32.8, 46.4]   |
| QRScore-8B-LME-TRAIN         | 91.1%   | **25.0%** (Δ66.1) | [18.8, 31.3]   |
| QRScore-8B-NQ-TRAIN          | 91.1%   | 85.9% (Δ5.2)      | [80.7, 90.6]   |
| Random (3-seed mean)         | 91.1%   | 88.2% (Δ2.9)      | n/a            |

Replication on Qwen-2.5-7B-Instruct (n=241, K=16): QRScore-SEC drops 89.2% → 34.0%,
NQ-TRAIN holds at 88.4% — same paradigm-specificity signature.

The full statement of claims and metric definitions lives in
[`findings/FINDINGS.md`](findings/FINDINGS.md). It is the single authoritative
source for paper writing.

---

## Layout

```
paper/
├── README.md                      ← this file
├── INVENTORY.md                   ← every artifact → paper claim mapping
├── data_manifest.md               ← what was deliberately NOT copied (raw CSVs)
├── LICENSE                        ← MIT (upstream)
│
├── findings/
│   ├── FINDINGS.md                ← AUTHORITATIVE: 8 key claims, all metric defs,
│   │                                bootstrap CIs, full per-task tables
│   └── UPSTREAM_README_cleanup.md ← upstream README from icml2026_submission_cleanup
│
├── reference/
│   └── 2506.09944v2.pdf           ← Wu et al. — the QRRetriever paper we build on
│
├── results/                       ← experimental evidence, organized by model
│   ├── llama_3_1_8B_instruct/     ← PRIMARY MODEL (192 instances, full pipeline)
│   │   ├── ablation/              ← accuracy curves, per-task heatmaps, summary
│   │   ├── transfer/              ← 8×8×8 cross-task transfer matrix + heatmaps
│   │   ├── specificity/           ← on/off-target drop, specificity index, CIs
│   │   ├── jaccard/               ← cross-method head overlap (SEC vs LME vs NQ)
│   │   ├── tables/                ← master CSVs (accuracy, drop, CIs)
│   │   └── raw_results/           ← per-method *_results.json (full curves)
│   │
│   ├── qwen_2_5_7B_instruct/      ← REPLICATION (241 instances, full pipeline +
│   │                                Transfer-* per-source results, drop-vs-k panels)
│   ├── mistral_7B_instruct/       ← ROBUSTNESS (full pipeline)
│   ├── olmo_7B/                   ← OPEN-WEIGHTS REPLICATION
│   │   ├── base/                  ← OLMo-7B (head-similarity only)
│   │   └── instruct/              ← OLMo-7B-Instruct (head-similarity only)
│   ├── cross_model/               ← cross-model Jaccard summary (Llama vs Qwen
│   │                                vs Mistral vs OLMo top-K head sets)
│   │   ├── kfe/                   ← K-FE table reproduction + verification
│   │   │   ├── kfe_table.csv
│   │   │   ├── kfe_report.md
│   │   │   └── VERIFICATION.md    ← Task 5 PASS record
│   │   ├── cross_model_union_overlap.csv   ← Task 4 output
│   │   ├── cross_model_union_overlap_plot.png
│   │   └── cross_model_union_overlap_REPORT.md
│   ├── fragility_predictors/      ← Task 2 — predictive fragility check
│   │   ├── correlations.csv
│   │   ├── scatter_plots.png
│   │   └── REPORT.md              ← VERDICT: SHARP (|r_baseline|=0.093)
│   ├── layer_analysis/            ← Task 1 — layer-position analysis
│   │   ├── per_task_layer_histograms.png
│   │   ├── fragile_vs_robust_overlay.png
│   │   ├── layer_concentration_table.csv
│   │   ├── permutation_tests.csv
│   │   └── REPORT.md              ← VERDICT: NOT SIGNIFICANT (best p=0.107)
│   └── random_head_null/          ← Task 3 — GPU pending
│       └── REQUIRES_GPU.md        ← run instructions + four-tier deployment table
│
├── data/
│   └── niah_input/                ← 8 *_test.json files (re-fetched, ~15 MB)
│                                    used by Task 2 query→answer-distance metric
│
├── detection/                     ← head ranking JSONs (small, ~5MB total)
│   ├── llama_3_1_8B_instruct/     ← per-task + combined + topk slices
│   ├── qwen_2_5_7B_instruct/
│   ├── mistral_7B_instruct/
│   ├── olmo_7B/{base,instruct}/
│   └── external_rankings/         ← pre-computed LME and NQ rankings
│                                    (lme_TRAIN_{llama,qwen}.json, nq_TRAIN_*)
│
├── figures/
│   ├── main/                      ← curated set proposed for the paper body
│   │   ├── fig1_apples_to_apples_llama.{pdf,png}     (4-method apples-to-apples
│   │   │                                              drop curves on shared n=192)
│   │   ├── fig2_kfe_asymmetry.{pdf,png}              (K-FE: target-sens vs
│   │   │                                              source-eff scatter, 3 pairs)
│   │   ├── fig1_llama_accuracy_vs_knockout.png       (legacy; pre-reaggregation)
│   │   ├── fig2_llama_per_task_curves.png            (per-task sensitivity)
│   │   ├── fig3_llama_transfer_K16.png               (Experiment 2)
│   │   ├── fig4_llama_specificity_drop_panels.png
│   │   ├── fig5_qwen_accuracy_vs_knockout.png        (replication)
│   │   ├── fig6_qwen_head_similarity.png             (functional clusters)
│   │   └── fig7_cross_model_jaccard_summary.pdf      (cross-model)
│   └── cross_ablation_curves/     ← 64 source-target accuracy curves
│                                    (8×8 task pairs, appendix material)
│
├── scripts/                       ← analysis pipeline (small)
│   ├── data_prep/                 ← split_dataset.py, build_detection_data.py,
│   │                                build_niah_data.py
│   ├── detection/                 ← detect_qrhead.py, run_detection.sh
│   ├── evaluation/                ← run_ablation.py, plot_ablation.py,
│   │                                plot_transfer.py, verify_leakage.py,
│   │                                compute_confidence_intervals.py,
│   │                                compute_cross_method_overlap.py,
│   │                                plot_task_head_jaccard.py,
│   │                                plot_nq_ablation_heatmap.py,
│   │                                compute_kfe_correlations.py,
│   │                                compute_target_sensitivity.py,
│   │                                verify_kfe.{sh,py},
│   │                                random_head_null.{py,sh}     ← GPU
│   ├── analysis/                  ← post-hoc, CPU-only
│   │   ├── predictive_fragility.py     (Task 2)
│   │   ├── cross_model_union_overlap.py (Task 4)
│   │   └── layer_distribution.py       (Task 1)
│   └── figures/                   ← regenerate the figures in figures/main/
│       ├── make_fig1_apples_to_apples.py
│       └── make_fig2_kfe_asymmetry.py
│
├── src/
│   ├── qrretriever/               ← core package (attn_retriever.py,
│   │                                custom_modeling_llama.py, custom_cache.py,
│   │                                config.py, predefined_heads.py, configs/)
│   └── setup.py
│
└── assets/qrheadlogo.png
```

### Pre-submission analyses summary

Five additional analyses have been added on top of the original submission scaffolding to pre-empt reviewer objections. Four are CPU-only and have been run; one (Task 3 random-head null) requires a GPU box and is staged with full run instructions.

| # | Task | Status | Verdict |
| ---: | --- | --- | --- |
| 2 | Predictive fragility check | DONE (CPU) | **SHARP** — \|r_baseline\|=0.093, p=0.827. Fragility ≠ baseline difficulty. |
| 5 | K-FE table verification | DONE (CPU) | **PASS** — all 6 rows reproduce within EPS=0.05. |
| 4 | Cross-model union overlap | DONE (CPU) | **WEAKER THAN HYPOTHESISED** — lifts 0.0×–1.7× at K=16 (we'd hoped 5–10×). §4 pivots to per-model phenomenon. |
| 1 | Layer-position analysis | DONE (CPU) | **NOT SIGNIFICANT** — best perm p=0.107 (OLMo-base). §4 reports descriptive histograms only, no mechanistic claim. |
| 3 | Random-head null distribution | **GPU PENDING** | Script + four-tier wrapper committed. See [`results/random_head_null/REQUIRES_GPU.md`](results/random_head_null/REQUIRES_GPU.md). |

See [`INVENTORY.md`](INVENTORY.md) → "Pre-submission analyses" for the full file-level mapping and the contingency narrative each verdict implies for §4 of the paper.

---

## What was filtered out (and why)

The upstream repo is ~250 MB on `main` and >300 MB across all branches. We kept ~39 MB.

| Dropped                                                            | Why                                                                      |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `data/{haystack,train,test}_plan.csv`, `needles.csv`, `sections.csv` | 95 MB of raw SEC text — recoverable from EDGAR, not needed to write the paper |
| `data/long_context_detection_optionA/*.json`                        | training-instance dumps used only to *produce* head rankings             |
| `data/niah_input/*_test.json`                                       | NIAH evaluation instances; the per-instance results are already in `*_results.json` |
| `*_token_log.jsonl` (16 files, ~7 MB)                               | per-instance generation logs for debug; accuracy is already aggregated   |
| `cross_ablation_curves.zip`                                         | duplicate of the unzipped `cross_ablation_curves/` directory             |
| `results/detection/.../_inputs/combined_detection.generated.json`   | re-derivable raw detection inputs (96 MB for OLMo alone)                 |
| `.git/`, `.DS_Store`, `.gitattributes`                              | VCS / OS metadata                                                        |
| `Llama-3.1-8B-Instruct/` and `Qwen-2.5-7B-Instruct/` top-level dirs | the only useful files (`lme_TRAIN.json`, `nq_TRAIN.json`) were promoted to `detection/external_rankings/` |

If the paper needs anything that was filtered out, it is regenerable from the
upstream repo: <https://github.com/4n4ny4/qr_scoring>.

See [`data_manifest.md`](data_manifest.md) for exact file sizes and recovery paths.

---

## Branch provenance

| Upstream branch                          | What we pulled                                                                        |
| ---------------------------------------- | ------------------------------------------------------------------------------------- |
| `ananya/icml2026_submission_cleanup`     | **base** — Llama, Qwen, scripts, src, README, FINDINGS                                |
| `ananya/final_final`                     | OLMo-7B (base + instruct) head-similarity, `cross_model_jaccard_*` (csv + pdf)        |
| `mistral_exp`                            | full Mistral-7B-Instruct ablation pipeline (results/, detection/)                     |
| `main`                                   | `cross_ablation_curves/` (64 source-target accuracy PNGs + summary CSV), `2506.09944v2.pdf` |

We deliberately ignored: `ananya/cross_ablations`, `ananya/gamma_olmo`,
`ananya/qwen_exp`, `nqheads-courtlistener-llama` — these are predecessors of, or
duplicates of, content already in the four sources above.

---

## How to use this for the paper

1. **Read `findings/FINDINGS.md` first.** It defines the experiments, metrics,
   numbers, and bootstrap CIs you will quote.
2. **Pick your figures from `figures/main/`.** They are pre-curated and re-mappable:
   each one has its source under `results/<model>/` if you want to swap variants.
3. **Use `INVENTORY.md`** to trace any specific number in the paper back to the
   exact CSV/JSON/PNG you cite.
4. **The LaTeX template lives in `../icml2026/`.** Workshop limit is 4 pages
   (short paper) or 8 pages (long), excluding references and appendix. We
   recommend the short-paper format (4 pages) given the focused contribution.
5. **All page limits exclude appendix** — the 64 cross-task curves and per-task
   detail panels live in `figures/cross_ablation_curves/` and
   `results/qwen_2_5_7B_instruct/figures_drop_vs_k/`, ready for a reproducibility
   appendix.

---

## Workshop fit (per CFP)

The CFP at <https://mechinterpworkshop.com/cfp/> explicitly welcomes:

- **Understanding model internals** — "How are {beliefs, personas, world models, **reasoning processes**, implicit goals} represented?" — this paper investigates how a *retrieval mechanism* is represented across the head population.
- **Methods for mechanistic discovery** — we evaluate the QRScore detection method's claims with controlled ablations and a random baseline.
- **Rigorous negative results** — Key Finding 4 is essentially a negative result for *task-specific* head circuits at the K we test.
- **Critiques or compelling failed replications of past work** — our paradigm-specificity finding qualifies the original QRRetriever claim.

Strong-empirical criteria the CFP demands and we already meet:

- **Falsifiable hypotheses, with evidence both for and against** — we state head-importance hypotheses up front and show where they hold (within span-extraction) and where they fail (cross-paradigm).
- **Clear practical benefits over baselines** — random ablation is the explicit baseline, with 3-seed averaging and bootstrap CIs.
- **Documented strengths and weaknesses** — Key Finding 3a explicitly flags the K=64 anomaly as noise; Key Finding 6 flags the per-task detections that fail to isolate.
- **Open-source code, models, prompts, data** — upstream is MIT-licensed; we ship all rankings, scripts, and per-instance accuracy curves.

The submission is **non-archival**, **double-blind** (search the manuscript for
GitHub user `4n4ny4` and contributor names before submission), and converts to
the ICML camera-ready format on acceptance. Reciprocal review is required (one
volunteer reviewer per submission).
