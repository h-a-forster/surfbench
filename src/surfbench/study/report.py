"""Aggregate finished units into the leaderboard with intervals and multiple-testing control.

Specification: ``docs/DESIGN.md`` section 6.4. Writes ``summary.json`` and
``leaderboard.md`` under the study's ``results_dir``. Neither file contains quotes, fitted
parameters, implied-volatility levels or absolute paths: only error and arbitrage
statistics, counts, hashes and calm/stress labels.
"""

from __future__ import annotations

import itertools
import json
import math
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from surfbench import __version__
from surfbench.checkpoint import UnitStore, atomic_write_bytes, atomic_write_json
from surfbench.quotes import FilterConfig
from surfbench.stats import (
    bootstrap_mean,
    bootstrap_ratio,
    holm,
    rank_probabilities,
    sign_flip_test,
)
from surfbench.study.runner import (
    StudyConfig,
    check_run_matches,
    expected_unit_ids,
    load_study,
)

__all__ = ["ACCURACY_METRICS", "build_summary", "render_leaderboard", "write_report"]

ACCURACY_METRICS = (
    "cv_rmse_iv",
    "cv_mae_iv",
    "cv_inside_spread",
    "cv_rms_spread_units",
    "wing_rmse_iv",
    "fit_rmse_iv",
    "next_rmse_iv",
    "next_rmse_iv_common",
)
ARB_SHARES = (
    ("bfly_viol_wide", "n_bfly_wide", "n_slices_checked"),
    ("bfly_viol_inrange", "n_bfly_inrange", "n_slices_checked"),
    ("price_viol_wide", "n_price_wide", "n_slices_checked"),
    ("cal_viol", "n_cal", "n_pairs"),
)


def _val(x: Any) -> float:
    return float(x) if isinstance(x, int | float) and x is not None else math.nan


def _finite(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def _ok_units(all_units: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    units = [
        u
        for u in all_units.values()
        if u.get("status") == "ok" and not u.get("empty") and u.get("models")
    ]
    return sorted(units, key=lambda u: (u["date"], u["underlying"]))


def _metric_matrix(units: list[dict[str, Any]], models: Sequence[str], metric: str) -> np.ndarray:
    return np.array([[_val(u["models"][m].get(metric)) for m in models] for u in units])


def _stress_labels(units: list[dict[str, Any]], quantile: float) -> dict[str, str]:
    """Stress = BTC 30-day ATM IV in the top quantile of the sample dates (BTC drives ETH)."""
    btc = {u["date"]: _val(u.get("atm30_iv")) for u in units if u["underlying"] == "BTC"}
    vals = np.array([v for v in btc.values() if math.isfinite(v)])
    if vals.size == 0:
        return {}
    thr = float(np.quantile(vals, quantile))
    return {d: ("stress" if v >= thr else "calm") for d, v in btc.items() if math.isfinite(v)}


def _regime_diff(a: np.ndarray, b: np.ndarray, n_boot: int, alpha: float, seed: int) -> dict:
    """Mean(a) - mean(b) with independent bootstrap resamples of the two groups."""
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return {
            "estimate": None,
            "lo": None,
            "hi": None,
            "n_stress": int(a.size),
            "n_calm": int(b.size),
        }
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    mb = b[rng.integers(0, b.size, (n_boot, b.size))].mean(axis=1)
    lo, hi = np.quantile(ma - mb, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(a.mean() - b.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "n_stress": int(a.size),
        "n_calm": int(b.size),
    }


def build_summary(cfg: StudyConfig) -> dict[str, Any]:
    rep = cfg.raw.get("report", {})
    n_boot = int(rep.get("n_boot", 10_000))
    n_perm = int(rep.get("n_perm", 100_000))
    alpha = float(rep.get("alpha", 0.05))
    blocks = [int(b) for b in rep.get("block_lengths", [1, 3])]
    primary = str(rep.get("primary_metric", "cv_rmse_iv"))
    seed = cfg.seed
    models = list(cfg.raw["fit"]["models"])
    store = UnitStore(cfg.out_dir)
    all_units = store.load_all()
    wanted = expected_unit_ids(cfg)
    if wanted is not None:  # units of dates no longer in the config are left out
        all_units = {k: u for k, u in all_units.items() if k in wanted}
    units = _ok_units(all_units)
    run_info_p = cfg.out_dir / "run_info.json"
    run_info = json.loads(run_info_p.read_text(encoding="utf-8")) if run_info_p.exists() else {}
    filt = FilterConfig.load(cfg.filter_path)
    labels = _stress_labels(units, float(rep.get("stress_quantile", 0.75)))

    summary: dict[str, Any] = {
        "study": cfg.name,
        "source": cfg.source,
        "models": models,
        "primary_metric": primary,
        "provenance": {
            "package_version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "seed": seed,
            "n_boot": n_boot,
            "n_perm": n_perm,
            "alpha": alpha,
            "block_lengths": blocks,
            "filter_version": filt.version,
            "filter_sha256": filt.sha256,
            "config_sha256": cfg.sha256,
            "horizon_min": cfg.raw["data"].get("horizon_min"),
            "t0": cfg.raw["data"].get("t0", "00:01:00"),
            "workers": run_info.get("workers"),
            "last_run_wall_s": _finite(_val(run_info.get("last_run_wall_s"))),
            "unit_compute_s": float(sum(_val(u.get("elapsed_s")) for u in units)),
        },
        "dates_missing": run_info.get("missing", []),
        "units_error": sorted(k for k, u in all_units.items() if u.get("status") == "error"),
        "units_empty": sorted(k for k, u in all_units.items() if u.get("empty")),
        "snapshots": sorted(
            {
                (u["date"], name, sha)
                for u in units
                for name, sha in u.get("snapshot_sha256", {}).items()
            }
        ),
        "regimes": labels,
        "by_underlying": {},
    }
    summary["snapshots"] = [{"date": d, "file": f, "sha256": s} for d, f, s in summary["snapshots"]]

    for und in sorted({u["underlying"] for u in units}):
        uu = [u for u in units if u["underlying"] == und]
        dates = [u["date"] for u in uu]
        block: dict[str, Any] = {
            "n_dates": len(uu),
            "dates": dates,
            "n_slices": int(sum(u["n_slices"] for u in uu)),
            "n_quotes": int(sum(u["n_quotes"] for u in uu)),
            "n_common_slices": int(sum(u.get("n_common_slices", 0) for u in uu)),
            "n_common_quotes": int(sum(u.get("n_common_quotes", 0) for u in uu)),
            "n_next_quotes": int(sum(u.get("n_next_quotes", 0) for u in uu)),
            "n_next_common_quotes": int(sum(u.get("n_next_common_quotes", 0) for u in uu)),
            "metrics": {},
            "arbitrage": {},
            "failures": {},
        }
        for metric in ACCURACY_METRICS:
            mat = _metric_matrix(uu, models, metric)
            block["metrics"][metric] = {
                m: bootstrap_mean(mat[:, i], n_boot=n_boot, alpha=alpha, seed=seed).as_dict()
                for i, m in enumerate(models)
            }
        pers = np.array([_val(u.get("persistence_rmse_iv")) for u in uu])
        block["metrics"]["persistence_rmse_iv"] = bootstrap_mean(
            pers, n_boot=n_boot, alpha=alpha, seed=seed
        ).as_dict()
        # Robustness: moving-block intervals on the primary metric.
        prim = _metric_matrix(uu, models, primary)
        block["primary_block_intervals"] = {
            str(b): {
                m: bootstrap_mean(
                    prim[:, i], n_boot=n_boot, alpha=alpha, block_length=b, seed=seed
                ).as_dict()
                for i, m in enumerate(models)
            }
            for b in blocks
        }
        for name, num, den in ARB_SHARES:
            block["arbitrage"][name] = {}
            for m in models:
                nn = np.array([_val(u["models"][m].get(num)) for u in uu])
                dd = np.array([_val(u["models"][m].get(den)) for u in uu])
                r = bootstrap_ratio(nn, dd, n_boot=n_boot, alpha=alpha, seed=seed).as_dict()
                r["violations"] = int(np.nansum(nn))
                r["checked"] = int(np.nansum(dd))
                block["arbitrage"][name][m] = r
        for m in models:
            block["failures"][m] = {
                "n_failed_slice_fits": int(
                    sum(u["models"][m].get("n_failed_slices", 0) for u in uu)
                ),
                "fit_time_s_mean": float(
                    np.mean([_val(u["models"][m].get("fit_time_s")) for u in uu])
                ),
            }
        # Rank probabilities on the common set of dates.
        fin = np.all(np.isfinite(prim), axis=1)
        block["rank_prob_lowest_" + primary] = (
            dict(
                zip(
                    models,
                    rank_probabilities(prim[fin], n_boot=n_boot, seed=seed).tolist(),
                    strict=True,
                )
            )
            if fin.any()
            else {}
        )
        # Regimes (exploratory).
        reg = np.array([labels.get(d, "unknown") for d in dates])
        block["regime_counts"] = {r: int(np.sum(reg == r)) for r in ("calm", "stress", "unknown")}
        block["by_regime"] = {
            r: {
                m: bootstrap_mean(
                    prim[reg == r, i], n_boot=n_boot, alpha=alpha, seed=seed
                ).as_dict()
                for i, m in enumerate(models)
            }
            for r in ("calm", "stress")
        }
        # Synthetic studies: primary metric by true generating family.
        truths = sorted({str(u.get("truth")) for u in uu if u.get("truth")})
        if truths:
            tr = np.array([str(u.get("truth")) for u in uu])
            block["by_truth"] = {
                t: {
                    m: bootstrap_mean(
                        prim[tr == t, i], n_boot=n_boot, alpha=alpha, seed=seed
                    ).as_dict()
                    for i, m in enumerate(models)
                }
                for t in truths
            }
        summary["by_underlying"][und] = block

    # Primary family: all pairwise differences, both underlyings, Holm across the family.
    tests = []
    for und in summary["by_underlying"]:
        uu = [u for u in units if u["underlying"] == und]
        prim = _metric_matrix(uu, models, primary)
        reg = np.array([labels.get(u["date"], "unknown") for u in uu])
        for (ia, a), (ib, b) in itertools.combinations(enumerate(models), 2):
            d = prim[:, ia] - prim[:, ib]
            keep = np.isfinite(d)
            p = sign_flip_test(d[keep], n_perm=n_perm, seed=seed)
            ci = bootstrap_mean(d[keep], n_boot=n_boot, alpha=alpha, seed=seed)
            tests.append(
                {
                    "underlying": und,
                    "a": a,
                    "b": b,
                    "mean_diff": _finite(ci.estimate),
                    "lo": _finite(ci.lo),
                    "hi": _finite(ci.hi),
                    "n": int(keep.sum()),
                    "p": p,
                    "regime_diff": _regime_diff(
                        d[keep & (reg == "stress")],
                        d[keep & (reg == "calm")],
                        n_boot,
                        alpha,
                        seed,
                    ),
                }
            )
    if tests:
        adj = holm([t["p"] for t in tests])
        for t, pa in zip(tests, adj, strict=True):
            t["p_holm"] = float(pa)
            t["reject_holm"] = bool(pa <= alpha)
    summary["pairwise_primary"] = tests
    return summary


def _fmt_iv(d: dict[str, Any], digits: int = 3, pct: bool = False) -> str:
    est, lo, hi = d.get("estimate"), d.get("lo"), d.get("hi")
    if est is None or not math.isfinite(est):
        return "n/a"
    f = 100.0 if pct else 1.0
    s = f"{est * f:.{digits}f}"
    if lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi):
        s += f" [{lo * f:.{digits}f}, {hi * f:.{digits}f}]"
    return s


def _fmt_p(p: float) -> str:
    if not math.isfinite(p):
        return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.4f}"


def render_leaderboard(summary: dict[str, Any]) -> str:
    models = summary["models"]
    prov = summary["provenance"]
    lines = [
        f"# Leaderboard: {summary['study']}",
        "",
        f"Source: `{summary['source']}`. Filter v{prov['filter_version']} "
        f"(sha256 `{prov['filter_sha256'][:12]}`). surfbench {prov['package_version']}, "
        f"Python {prov['python']}, numpy {prov['numpy']}, scipy {prov['scipy']}. Seed "
        f"{prov['seed']}; {prov['n_boot']} bootstrap resamples (dates resampled), "
        f"{prov['n_perm']} sign flips; next-snapshot horizon {prov['horizon_min']} min.",
        "",
        "Means over dates with 95% percentile bootstrap intervals. Errors in vol points; "
        "shares in percent. Arbitrage shares are pooled over slices (pairs) with a "
        "date-cluster bootstrap interval.",
        "",
    ]
    for und, b in summary["by_underlying"].items():
        lines += [
            f"## {und}",
            "",
            f"{b['n_dates']} dates, {b['n_slices']} slices, {b['n_quotes']} quotes "
            f"({b['n_common_slices']} slices / {b['n_common_quotes']} quotes where every model "
            f"fit every fold); next snapshot: {b['n_next_quotes']} quotes, "
            f"{b['n_next_common_quotes']} instruments filtered at both times.",
            "",
            "| model | CV RMSE | inside spread % | wing RMSE | next RMSE (common) | "
            "butterfly % (wide) | butterfly % (in range) | price check % | calendar % |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        mt, ar = b["metrics"], b["arbitrage"]
        for m in models:
            lines.append(
                f"| {m} | {_fmt_iv(mt['cv_rmse_iv'][m])} | "
                f"{_fmt_iv(mt['cv_inside_spread'][m], 1, pct=True)} | "
                f"{_fmt_iv(mt['wing_rmse_iv'][m])} | {_fmt_iv(mt['next_rmse_iv_common'][m])} | "
                f"{_fmt_iv(ar['bfly_viol_wide'][m], 1, pct=True)} | "
                f"{_fmt_iv(ar['bfly_viol_inrange'][m], 1, pct=True)} | "
                f"{_fmt_iv(ar['price_viol_wide'][m], 1, pct=True)} | "
                f"{_fmt_iv(ar['cal_viol'][m], 1, pct=True)} |"
            )
        lines += [
            "",
            f"Quote persistence (next snapshot, common instruments): "
            f"{_fmt_iv(mt['persistence_rmse_iv'])} vol points.",
            "",
            "P(lowest mean CV RMSE): "
            + ", ".join(
                f"{m} {p:.2f}"
                for m, p in b.get(f"rank_prob_lowest_{summary['primary_metric']}", {}).items()
            ),
            "",
        ]
        if b.get("by_truth"):
            truths = list(b["by_truth"])
            lines += [
                "CV RMSE by true generating family (synthetic data):",
                "",
                "| model | " + " | ".join(truths) + " |",
                "|---|" + "---|" * len(truths),
            ]
            for m in models:
                lines.append(
                    f"| {m} | " + " | ".join(_fmt_iv(b["by_truth"][t][m]) for t in truths) + " |"
                )
            lines.append("")
        rc = b["regime_counts"]
        if rc.get("stress", 0) and rc.get("calm", 0):
            lines += [
                f"Exploratory regimes (stress = BTC 30-day ATM IV in the top quartile): "
                f"{rc['calm']} calm, {rc['stress']} stress dates. CV RMSE by regime:",
                "",
                "| model | calm | stress |",
                "|---|---|---|",
            ]
            for m in models:
                lines.append(
                    f"| {m} | {_fmt_iv(b['by_regime']['calm'][m])} | "
                    f"{_fmt_iv(b['by_regime']['stress'][m])} |"
                )
            lines.append("")
    lines += [
        "## Pairwise comparisons on CV RMSE (primary family, Holm-adjusted)",
        "",
        "Mean of date-level differences `a - b` (negative: `a` is more accurate), bootstrap "
        "95% interval, two-sided sign-flip p-value and Holm-adjusted p over all "
        f"{len(summary['pairwise_primary'])} tests.",
        "",
        "| underlying | a | b | mean diff | 95% CI | n | p | p (Holm) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in summary["pairwise_primary"]:
        md = "n/a" if t["mean_diff"] is None else f"{t['mean_diff']:+.3f}"
        ci = "n/a" if t["lo"] is None else f"[{t['lo']:+.3f}, {t['hi']:+.3f}]"
        lines.append(
            f"| {t['underlying']} | {t['a']} | {t['b']} | {md} | {ci} | {t['n']} | "
            f"{_fmt_p(t['p'])} | {_fmt_p(t.get('p_holm', float('nan')))} |"
        )
    missing = summary.get("dates_missing", [])
    lines += [
        "",
        f"Dates missing: {len(missing)}"
        + (": " + ", ".join(f"{m['date']} ({m['reason']})" for m in missing) if missing else "")
        + ". Units with errors: "
        + (", ".join(summary["units_error"]) or "none")
        + ".",
        "",
    ]
    return "\n".join(lines)


def write_report(config: Path, root: Path | None = None) -> Path:
    cfg = load_study(config, root)
    check_run_matches(cfg)
    summary = build_summary(cfg)
    out = cfg.results_dir
    atomic_write_json(out / "summary.json", summary)
    atomic_write_bytes(out / "leaderboard.md", render_leaderboard(summary).encode("utf-8"))
    return out
