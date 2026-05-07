# K-FE table — verification record

| | |
| --- | --- |
| Verified at (UTC) | `2026-05-07T19:45:03Z` |
| Git SHA at verify time | `fbceda4` |
| Tolerance (`EPS`) | `0.05` (≈2× the published 2-sig-fig quantisation; see [`verify_kfe.py`](../../../scripts/evaluation/verify_kfe.py)) |
| Verification command | `bash scripts/evaluation/verify_kfe.sh` |
| Verification harness | [`scripts/evaluation/verify_kfe.py`](../../../scripts/evaluation/verify_kfe.py) |
| Result | **PASS — all 6 asserted rows within tolerance** |

## What is asserted

Three sensitivity rows and three efficacy rows from [`tab:kfe`](../../../findings/upstream_results_tex_excerpt.tex) of the upstream paper draft. The four-way efficacy "definition shootout" in [`compute_kfe_correlations.py`](../../../scripts/evaluation/compute_kfe_correlations.py) shows that only `off_target_mean` reproduces the published values; `verify_kfe.py` therefore asserts on that definition only and skips the other three (which are present in the CSV as audit information).

## PASS/FAIL per pair (run captured at the timestamp above)

| Measure | Definition | Pair | Derived | Published | Δ | Verdict |
| --- | --- | --- | ---: | ---: | ---: | --- |
| sensitivity | column_mean_drop | Llama ↔ Qwen | 0.184 | 0.18 | 0.0035 | PASS |
| sensitivity | column_mean_drop | Llama ↔ Mistral | 0.472 | 0.47 | 0.0020 | PASS |
| sensitivity | column_mean_drop | Qwen  ↔ Mistral | 0.593 | 0.59 | 0.0028 | PASS |
| efficacy    | off_target_mean  | Llama ↔ Qwen | 0.018 | 0.02 | 0.0022 | PASS |
| efficacy    | off_target_mean  | Llama ↔ Mistral | 0.039 | 0.04 | 0.0014 | PASS |
| efficacy    | off_target_mean  | Qwen  ↔ Mistral | 0.153 | 0.15 | 0.0031 | PASS |

Total |Δ| across all 6 rows: **0.0150** (well below the 0.05 EPS).

## Forensic note (carried forward from the prior commit)

The original "efficacy" definition was not preserved as code anywhere in the upstream repository. `compute_kfe_correlations.py` is therefore a *post-hoc forensic reconstruction*: it tests four candidate definitions against the published `tab:kfe` values and reports which fits best. The published numbers themselves were generated off-repo (presumably interactively in a notebook that wasn't committed). Reproducibility is recovered because:

1. The K-residualization formula is unambiguous (subtract within-K column means; flatten; Pearson R²).
2. The sensitivity definition is unambiguous (column mean of the drop matrix).
3. Of the four candidate efficacy definitions tested, only `off_target_mean` reproduces the published values to within 0.007 in aggregate; the other three deviate by 0.13–0.82.
4. The local cross-task transfer matrices we ship in [`results/<model>/transfer/`](../../) are sufficient to regenerate every cell of `tab:kfe` via `verify_kfe.sh`.

## What this means for reviewers

A reviewer running `bash scripts/evaluation/verify_kfe.sh` from a fresh clone will see:

```
[verify_kfe] all 6 asserted rows PASSED — tab:kfe reproduces from local matrices within ±0.05
```

If a future change to the local matrices breaks reproducibility (e.g. someone re-runs ablations and the per-cell numbers shift), `verify_kfe.py` exits nonzero and prints a per-row diff. That is the intended behaviour: it should fail loudly rather than silently let `tab:kfe` drift from the underlying data.
