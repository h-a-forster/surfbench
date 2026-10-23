"""Calibrator quality check: our per-slice calibrators against a global optimiser.

For random noisy synthetic slices, compares the common objective ``L`` reached by our raw SVI
and SABR calibrators with the best of scipy's differential evolution (with polish) under the
same constraints. A calibrator that lands in a worse local minimum shows up as
``L_ours > L_global``. Synthetic data only, so the output can be committed.

Run: ``uv run python scripts/calibrator_check.py --n 60 --out results/calibrator-check``
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from surfbench import synthetic
from surfbench.models.base import iv_residuals
from surfbench.models.sabr import SabrModel, hagan_lognormal_vol
from surfbench.models.svi import SviModel, svi_constraints_ok, svi_raw


def _svi_global(k, iv, scale, T, seed):
    wmax = float((iv * iv * T).max())

    def loss(q):
        a, b, rho, m, s = q
        pen = 0.0
        minvar = a + b * s * np.sqrt(max(1 - rho * rho, 0.0))
        if minvar < 0:
            pen += 1e6 * (-minvar + 1e-3)
        if b * (1 + abs(rho)) > 2:
            pen += 1e6 * (b * (1 + abs(rho)) - 2 + 1e-3)
        r = iv_residuals(svi_raw(k, a, b, rho, m, s)[0], T, iv, scale)
        return float(r @ r) + pen

    bounds = [
        (-wmax - 0.1, wmax + 0.1),
        (0, 2),
        (-0.999, 0.999),
        (k.min() - 1, k.max() + 1),
        (1e-4, 2),
    ]
    res = differential_evolution(
        loss, bounds, seed=seed, tol=1e-12, maxiter=3000, popsize=30, polish=True
    )
    ok = svi_constraints_ok(*res.x)
    return float(res.fun) if ok else float("nan")


def _sabr_global(k, iv, scale, T, seed):
    def loss(q):
        r = (hagan_lognormal_vol(k, T, *q) - iv) / scale
        r = np.where(np.isfinite(r), r, 1e3)
        return float(r @ r)

    bounds = [(1e-4, 10), (-0.999, 0.999), (1e-4, 10)]
    res = differential_evolution(
        loss, bounds, seed=seed, tol=1e-12, maxiter=3000, popsize=30, polish=True
    )
    return float(res.fun)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="number of random surfaces")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.perf_counter()
    for i in range(args.n):
        fam = ("essvi", "svi", "sabr")[i % 3]
        surf = synthetic.random_surface(fam, rng, arbitrage_free=True)
        snap = synthetic.make_snapshot(surf, noise=0.005, spread=0.02, seed=i)
        d = snap.data
        j = int(rng.integers(0, d.n_slices))
        k, iv, sc, T = d.k[j], d.iv_mid[j], d.scale(j), d.expiries[j]
        one = type(d)((T,), (k,), (iv,), (d.half_spread[j],), seed=i)
        for name, model, glob in (
            ("svi", SviModel(), _svi_global),
            ("sabr", SabrModel(), _sabr_global),
        ):
            fit = model.fit(one)
            r = iv_residuals(fit.w(0, k), T, iv, sc)
            ours = float(r @ r)
            best = glob(k, iv, sc, T, args.seed + i)
            rows.append({"truth": fam, "model": name, "L_ours": ours, "L_global": best})
    out: dict[str, object] = {"n_surfaces": args.n, "seed": args.seed, "models": {}}
    for name in ("svi", "sabr"):
        rr = [r for r in rows if r["model"] == name and np.isfinite(r["L_global"])]
        ours = np.array([r["L_ours"] for r in rr])
        glob = np.array([r["L_global"] for r in rr])
        rel = (ours - glob) / np.maximum(glob, 1e-12)
        out["models"][name] = {
            "n": len(rr),
            "share_ours_not_worse": float(np.mean(rel <= 1e-6)),
            "share_ours_within_1pct": float(np.mean(rel <= 0.01)),
            "median_rel_gap": float(np.median(rel)),
            "max_rel_gap": float(np.max(rel)),
            "share_ours_better_by_1pct": float(np.mean(rel < -0.01)),
        }
    out["wall_s"] = round(time.perf_counter() - t0, 1)
    text = json.dumps(out, indent=2)
    print(text)
    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
