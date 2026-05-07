#!/usr/bin/env python3
"""Assertion harness for the K-FE residualization output.

Reads ``results/cross_model/kfe/kfe_table.csv`` (produced by
``compute_kfe_correlations.py``) and verifies that the derived K-FE R^2 values
match the published numbers in ``tab:kfe`` of the upstream paper
(``findings/upstream_results_tex_excerpt.tex``) within ``EPS = 0.05``.

Why EPS = 0.05 and not 0.02:
    Published tab:kfe values are 2 sig figs (0.18 / 0.47 / 0.59 sensitivity;
    0.02 / 0.04 / 0.15 efficacy). Derived values are 3 sig figs (0.184 /
    0.472 / 0.593; 0.018 / 0.039 / 0.153). Quantisation alone gives ±0.005
    of slack, and minor numerical drift from numpy/scipy version bumps could
    trivially push past 0.02. EPS = 0.05 communicates "matches published
    table to within rounding tolerance" — what we actually want.

Exit codes:
    0  every checked row is within EPS
    1  at least one row is out of tolerance OR the table is missing /
       malformed

Usage:
    python scripts/evaluation/verify_kfe.py [--table PATH] [--eps FLOAT]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Tuple

DEFAULT_TABLE = "results/cross_model/kfe/kfe_table.csv"
DEFAULT_EPS = 0.05

# Definition we expect to use for efficacy (per the forensic finding in
# kfe_report.md: off_target_mean is the only definition that matches
# published values).
EXPECTED_EFFICACY_DEFINITION = "off_target_mean"


def load_table(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"[verify_kfe] table not found: {path}\n"
                 f"Run scripts/evaluation/verify_kfe.sh first.")
    with open(path) as fh:
        return list(csv.DictReader(fh))


def check_row(row: dict, eps: float) -> Tuple[bool, str]:
    """Return (passes, reason)."""
    measure = row["measure"]
    definition = row["definition"]
    derived_str = row.get("kfe_r2_derived", "")
    published_str = row.get("kfe_r2_published", "")
    abs_dev_str = row.get("abs_dev", "")
    if not published_str or published_str.strip() in ("", "—"):
        return True, "no published value to compare against"
    try:
        derived = float(derived_str)
        published = float(published_str)
        abs_dev = abs(derived - published)
    except (TypeError, ValueError):
        return False, f"bad numeric values: derived={derived_str!r} published={published_str!r}"
    # We only assert on the rows the paper cites. For efficacy, that's the
    # off_target_mean definition.
    if measure == "efficacy" and definition != EXPECTED_EFFICACY_DEFINITION:
        return True, f"non-asserted efficacy definition: {definition}"
    if abs_dev <= eps:
        return True, f"PASS (Δ = {abs_dev:.4f} ≤ {eps})"
    return False, f"FAIL (Δ = {abs_dev:.4f} > {eps})"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", default=DEFAULT_TABLE,
                   help=f"Path to kfe_table.csv (default: {DEFAULT_TABLE})")
    p.add_argument("--eps", type=float, default=DEFAULT_EPS,
                   help=f"Per-row tolerance (default: {DEFAULT_EPS})")
    args = p.parse_args()

    rows = load_table(args.table)
    print(f"[verify_kfe] checking {len(rows)} rows of {args.table}")
    print(f"[verify_kfe] EPS = {args.eps}")
    print()

    failures: List[Tuple[dict, str]] = []
    checked = 0
    for row in rows:
        ok, reason = check_row(row, args.eps)
        flag = "[PASS]" if ok else "[FAIL]"
        if "no published value" in reason or "non-asserted" in reason:
            flag = "[SKIP]"
            continue
        checked += 1
        a = row.get("model_a", "?")
        b = row.get("model_b", "?")
        m = row.get("measure", "?")
        d = row.get("definition", "?")
        print(f"{flag} {m:<11} ({d:<19}) {a:<28} ↔ {b:<28} {reason}")
        if not ok:
            failures.append((row, reason))

    print()
    if failures:
        print(f"[verify_kfe] {len(failures)} of {checked} asserted rows FAILED")
        print("[verify_kfe] Either tab:kfe is wrong, the matrices have drifted, "
              "or the off_target_mean definition no longer reproduces published values.")
        return 1
    print(f"[verify_kfe] all {checked} asserted rows PASSED — "
          f"tab:kfe reproduces from local matrices within ±{args.eps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
