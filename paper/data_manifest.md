# Data manifest — what was filtered out

This file documents the upstream data files that were **deliberately not copied**
into this paper folder, so a reader can recover them from the upstream repo
(<https://github.com/4n4ny4/qr_scoring>) if the paper requires it.

## Why filter?

The upstream repo is dominated by raw text dumps and per-instance debug logs
that are **redundant** for paper writing:

- Raw SEC filings can be re-downloaded from EDGAR using `scripts/data_prep/`.
- The aggregated accuracy curves we keep are derived from the per-instance logs we drop.

We kept the *outputs* (head rankings, accuracy curves, statistical summaries)
rather than the *inputs* (raw SEC text, per-instance generations).

## Filtered files

### Raw SEC corpus (~95 MB)

| Upstream path                  | Size  | What it is |
| ------------------------------ | ----- | ---------- |
| `data/haystack_plan.csv`       | 43 MB | All SEC instances pre-split (8 tasks × 974 instances) |
| `data/train_plan.csv`          | 35 MB | 80% train split by filing |
| `data/test_plan.csv`           | 8 MB  | 20% test split by filing |
| `data/needles.csv`             | 15 MB | Extracted gold facts per filing |
| `data/sections.csv`            | 15 MB | Section-level text from filings |

**Recovery:** `git clone https://github.com/4n4ny4/qr_scoring.git && cd qr_scoring`
or rebuild via:

```bash
python scripts/data_prep/split_dataset.py
python scripts/data_prep/build_detection_data.py --chunk_words 400
python scripts/data_prep/build_niah_data.py --chunk_words 400
```

### Generated detection / NIAH instances

| Upstream path                                | What it is |
| -------------------------------------------- | ---------- |
| `data/long_context_detection_optionA/*.json` | Long-context training instances used to score heads (Step 5 input) |
| `data/niah_input/*_test.json`                | NIAH evaluation instances (Step 6+ input) |
| `results/detection/<model>/_inputs/combined_detection.generated.json` | Pre-tokenised detection inputs (96 MB for OLMo) |

**Recovery:** Re-run the build scripts above, or `git checkout <branch> -- data/`.

### Per-instance generation logs (~7 MB)

| Upstream path pattern                                       | What it is |
| ----------------------------------------------------------- | ---------- |
| `results/comparison_ablation/.../QRScore-SEC_token_log.jsonl`        | Raw token-level generations + answer extraction for every test instance, every K |
| `results/comparison_ablation/.../QRScore-8B-LME-TRAIN_token_log.jsonl` | " |
| `results/comparison_ablation/.../QRScore-8B-NQ-TRAIN_token_log.jsonl` | " |
| `results/comparison_ablation/.../Random-seed{42,123,456}_token_log.jsonl` | " |
| `results/comparison_ablation/.../Transfer-<task>_token_log.jsonl` (×8) | " |

These are useful for *post-hoc qualitative inspection of model failures* but
the per-instance accuracy already lives in the corresponding `*_results.json`
files we kept. If you want to inspect specific failure modes (e.g. where the
model produces a partial substring match), grep these files in upstream.

### Other dropped items

| Path                                  | Why                                                                |
| ------------------------------------- | ------------------------------------------------------------------ |
| `cross_ablation_curves.zip`           | duplicate of the unzipped `cross_ablation_curves/` directory we kept |
| `.DS_Store` (multiple)                | macOS metadata                                                     |
| `.git/`                               | VCS metadata                                                       |
| `Llama-3.1-8B-Instruct/{lme,nq}_TRAIN.json` (top-level) | promoted to `detection/external_rankings/` |
| `Qwen-2.5-7B-Instruct/{lme,nq}_TRAIN_qwen.json` (top-level) | promoted to `detection/external_rankings/` |

## What is the dataset, in one paragraph?

The 8 SEC tasks each ask the model to extract one fact from a long SEC 10-K
filing. The corpus is 974 training instances + 241 evaluation instances, with
a document-level (filename-based) 80/20 split that prevents leakage. The 8
tasks are: `registrant_name`, `headquarters_city`, `headquarters_state`,
`incorporation_state`, `incorporation_year`, `employees_count_total`,
`ceo_lastname`, `holder_record_amount`. For ablation evaluation, the test set
is capped at 24 instances per task (`--max_instances_per_task 24`) for balanced
comparison, giving 192 evaluated instances on Llama and 241 on Qwen
(uncapped).

## Reproducibility

The upstream pipeline (Steps 1–8 in `findings/UPSTREAM_README_cleanup.md`) is
fully reproducible from a clean clone, given a CUDA GPU with HuggingFace access
to `meta-llama/Llama-3.1-8B-Instruct`. A smoke test takes minutes; a full run
takes hours per model.

```bash
pip install -e src/  # uses src/setup.py
bash scripts/detection/run_detection.sh
python scripts/evaluation/run_ablation.py \
  --niah_dir data/niah_input \
  --output_dir results/comparison_ablation \
  --max_instances_per_task 24 \
  --max_context_tokens 8192 \
  --knockout_sizes 0 8 16 32 48 64 96 128 \
  --enable_cross_task_transfer \
  --include_random_baselines \
  --methods QRScore-SEC QRScore-8B-LME-TRAIN QRScore-8B-NQ-TRAIN
```
