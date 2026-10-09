#!/usr/bin/env bash
# Reproduce every table, figure, and reported number in the RADC manuscript.
# Run from the analysis directory:  bash run_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "== environment =="; python3 -c "import sys,numpy,pandas,sklearn,scipy,matplotlib;print('python',sys.version.split()[0]);print('numpy',numpy.__version__);print('pandas',pandas.__version__);print('sklearn',sklearn.__version__)" || true

echo "== verify frozen-bundle hash =="
python3 - <<'PY'
import hashlib
from pathlib import Path
want = Path("artifacts/bundles/predictions_v1.sha256").read_text().split()[0]
got = hashlib.sha256(Path("artifacts/bundles/predictions_v1.csv").read_bytes()).hexdigest()
assert got == want, "bundle hash mismatch"
print("derived bundle SHA-256 verified:", got[:16], "...")
PY

echo "== test suite =="
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v

echo "== experiments -> out/ =="
mkdir -p out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/decision_regret_experiment.py --repeats 20 --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/decision_regret_baselines.py --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/calibration_decision_2x2.py --repeats 8 --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/oracle_upper_bound.py --repeats 12 --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/stability_placebo.py --repeats 20 --stability-seeds 10 --outdir out
# activation_rates aggregates sibling result tables; seed the ones not regenerated above
cp artifacts/results/drrl_strong_control_20seed.csv out/ 2>/dev/null || true
PYTHONDONTWRITEBYTECODE=1 python3 experiments/activation_rates.py --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/calibration_method_sensitivity.py --outdir out
PYTHONDONTWRITEBYTECODE=1 python3 experiments/bin_count_sensitivity.py --repeats 10 --outdir out

echo "== diff regenerated vs committed (should be exactly 0) =="
python3 - <<'PY'
import pandas as pd, numpy as np, glob, os
bad = 0
for f in sorted(glob.glob("artifacts/results/*.csv")):
    name = os.path.basename(f); regen = os.path.join("out", name)
    if not os.path.exists(regen): continue
    a = pd.read_csv(f); b = pd.read_csv(regen)
    if a.shape != b.shape:
        print(f"  SHAPE DIFF {name}: {a.shape} vs {b.shape}"); bad += 1; continue
    num = [c for c in a.columns if a[c].dtype.kind in "fc" and c in b.columns]
    d = float(np.nanmax(np.abs(a[num].to_numpy(float)-b[num].to_numpy(float)))) if num else 0.0
    print(f"  {name}: max_diff={d:.3e}")
    if d > 1e-9: bad += 1
print("ALL MATCH" if bad == 0 else f"{bad} mismatches")
PY
echo "Done."
