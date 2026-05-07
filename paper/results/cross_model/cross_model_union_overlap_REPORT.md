# Cross-model union-overlap report

## Method

For each model `M` and top-K threshold, we build the union
`U_M(K) = ∪_{t ∈ 8 SEC tasks} top-K heads detected for task t`.
For each pair of models we compute Jaccard `|U_a ∩ U_b| / |U_a ∪ U_b|`
and the overlap coefficient `|U_a ∩ U_b| / min(|U_a|, |U_b|)`,
plus an empirical random-baseline distribution from 1000 random
head-subset pairs sampled at the same sizes from each model's
head population.

Head populations: 1024 for Llama / Mistral / OLMo-Instruct, 784 for Qwen.

## Headline (K=16) — same-head-population pairs

| Pair | \|U_a\| | \|U_b\| | ∩ | Jaccard obs | Jaccard rand mean | Lift | p (one-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama ↔ Mistral | 37 | 33 | 1 | 0.014 | 0.017 | **0.8×** | 0.692 |
| Llama ↔ OLMo | 37 | 32 | 1 | 0.015 | 0.017 | **0.9×** | 0.695 |
| Mistral ↔ OLMo | 33 | 32 | 0 | 0.000 | 0.017 | **0.0×** | 1 |

## Headline (K=16) — cross-population pairs (Qwen vs others)

Reported using overlap coefficient `|U_a ∩ U_b| / min(|U_a|, |U_b|)`
because Jaccard is awkward across head populations of different size.

| Pair | \|U_a\| | \|U_b\| | ∩ | Overlap obs | Overlap rand mean | Lift | p (one-sided) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama ↔ Qwen | 37 | 40 | 3 | 0.081 | 0.051 | **1.6×** | 0.3 |
| Mistral ↔ Qwen | 33 | 40 | 2 | 0.061 | 0.051 | **1.2×** | 0.511 |
| OLMo ↔ Qwen | 32 | 40 | 1 | 0.031 | 0.052 | **0.6×** | 0.821 |

## Verdict template

If observed lifts at K=16 are ≥ 5×, the paper claim becomes
*'Cross-model retrieval-head-pool overlap is N× above random*
*expectation while per-task efficacy correlations stay below 0.15 R² —*
*heads are SHARED SUBSTRATE, DIFFERENTLY USED.'*

If observed lifts are ≤ 2×, the cross-model substrate claim weakens
and the §4 prose pivots to per-model phenomenon.

## Note on existing files in `results/cross_model/`

This analysis is **distinct from**:
- `cross_model_head_overlap.csv` (within-model SEC/LME/NQ comparison).
- `cross_model_jaccard_*.csv` (per-task within-model Jaccard).

All three live in the same directory because they all touch *some* notion
of head-set overlap; this one is the cross-model union comparison.

## All-K full table

See `cross_model_union_overlap.csv` for the K ∈ {8, 16, 32, 48, 64, 96, 128}
values for every pair.
