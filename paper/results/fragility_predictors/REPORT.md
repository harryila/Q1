# Predictive fragility check — REPORT

## Decision rule

| |r_baseline| | Verdict | Action |
| --- | --- | --- |
| < 0.4 | SHARP | Claim sharpens to 'fragility is not difficulty in disguise' |
| 0.4 ≤ x ≤ 0.7 | PARTIAL | Downgrade to 'partially decouples'; report in §5 limitations |
| > 0.7 | AT_RISK | Re-orient §3 around joint difficulty/fragility analysis |

Observed `|r_baseline| = 0.093` → **Verdict: SHARP** — fragility does not correlate with baseline accuracy. The structural-fragility claim survives.

## Per-task values

| Task | Fragility | Baseline acc | Q→A token-dist | Answer tok-len |
| --- | ---: | ---: | ---: | ---: |
| `ceo_lastname` | 0.431 | 0.912 | 9 | 1.0 |
| `employees_count_total` | 0.692 | 0.931 | 64 | 1.0 |
| `headquarters_city` | 0.336 | 0.962 | 148 | 1.0 |
| `headquarters_state` | 0.218 | 0.881 | 856 | 1.0 |
| `holder_record_amount` | 0.398 | 0.528 | 220 | 1.0 |
| `incorporation_state` | 0.415 | 0.976 | 288 | 1.0 |
| `incorporation_year` | 0.473 | 0.831 | 189 | 1.0 |
| `registrant_name` | 0.110 | 0.964 | 519 | 3.0 |

## Correlations vs fragility

| Predictor | Pearson r | 95% CI | Pearson p | Spearman ρ | Spearman p |
| --- | ---: | --- | ---: | ---: | ---: |
| baseline_accuracy | -0.093 | [-0.652, +0.331] | 0.827 | -0.214 | 0.61 |
| query_answer_distance | -0.720 | [-0.960, -0.266] | 0.044 | -0.714 | 0.0465 |
| answer_token_length | -0.639 | [-0.977, -0.502] | 0.0884 | -0.577 | 0.134 |

## Notes

- Fragility is `mean_M mean_{K∈[8, 16]} sensitivity_M[t][K]` with `sensitivity = mean_s drop[s][t][K]`. Computed on the 3 transfer matrices in `results/<model>/transfer/`.
- Baseline accuracy is `K=0` per-task accuracy from `QRScore-SEC_results.json`, averaged across the 3 models.
- Query→answer token distance is the median over instances of the minimum word-distance from any task-keyword's first occurrence to the answer's first occurrence in the haystack. Word-distance is a CPU-only proxy for LLM-token-distance (~1.3 tokens/word; relative ordering preserved).
- Answer token length is the median word count of `needle_value` per task.
- All p-values reported are from 2-sided Pearson/Spearman with `n = 8` tasks; bootstrap CI uses 1000 task-resamples (low power; treat CIs as descriptive, not confirmatory).

## Original (broken) predictor

The earlier draft proposed `Predictor 2 = min(needle_char_offset, total_chars - needle_char_offset)` (depth from haystack edges). Every instance in `data/niah_input/*_test.json` has `needle_position == "middle"` (we verified all 8 files), so this metric is constant across instances and tasks and has zero variance — it cannot correlate with anything. Replaced with query→answer token distance, which is the within-document retrieval-difficulty proxy that does vary with task structure.
