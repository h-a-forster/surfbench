"""Study runner: one checkpointed unit per (snapshot date, underlying).

Specification: ``docs/DESIGN.md`` section 6.2. Data sources (``[data].source``):

- ``tardis``: ``<cache_dir>/YYYY-MM-01/{snap_t0,snap_t1}.csv.gz`` written by ``data tardis``;
- ``deribit``: ``<cache_dir>/<stamp>/{snap_t0,snap_t1}.csv.gz`` written by ``data deribit``
  (development only; results from live snapshots are never committed);
- ``synthetic``: snapshots generated from known surfaces (validation and tests).
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import tomllib
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from surfbench import __version__, synthetic
from surfbench.arbitrage import ArbitrageConfig, diagnose
from surfbench.checkpoint import RunMismatchError, UnitStore, atomic_write_json, run_units
from surfbench.metrics import (
    fold_ids,
    fold_offsets,
    inside_spread,
    mae_volpt,
    rms_spread_units,
    rmse_volpt,
    wing_mask,
)
from surfbench.models import accepted, get_model
from surfbench.models.ssvi import atm_total_variance
from surfbench.quotes import (
    YEAR_SECONDS,
    FilterConfig,
    Slice,
    apply_filter,
    data_quality,
    file_sha256,
    merge_snapshots,
    read_snapshot,
    to_fit_input,
    write_snapshot,
)

__all__ = ["StudyConfig", "discover_units", "fit_table", "load_study", "run_study", "run_unit"]


def _norm_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


@dataclass(frozen=True)
class StudyConfig:
    path: Path
    root: Path
    raw: dict[str, Any]
    sha256: str

    @property
    def name(self) -> str:
        return str(self.raw["study"]["name"])

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @property
    def out_dir(self) -> Path:
        return self.resolve(self.raw["study"]["out_dir"])

    @property
    def results_dir(self) -> Path:
        return self.resolve(self.raw["study"]["results_dir"])

    @property
    def cache_dir(self) -> Path:
        return self.resolve(self.raw["data"]["cache_dir"])

    @property
    def filter_path(self) -> Path:
        return self.resolve(self.raw["fit"]["filter"])

    @property
    def seed(self) -> int:
        return int(self.raw["study"]["seed"])

    @property
    def source(self) -> str:
        return str(self.raw["data"].get("source", "tardis"))

    def arbitrage(self) -> ArbitrageConfig:
        a = self.raw.get("arbitrage", {})
        return ArbitrageConfig(
            grid_z=tuple(a.get("grid_z", (-4.0, 4.0))),  # type: ignore[arg-type]
            n_grid=int(a.get("n_grid", 401)),
            g_tol=float(a.get("g_tol", 1e-6)),
            calendar_tol=float(a.get("calendar_tol", 1e-8)),
        )


def load_study(path: Path, root: Path | None = None) -> StudyConfig:
    """Load a study config; relative paths in it resolve against ``root`` (default: cwd)."""
    path = Path(path)
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    return StudyConfig(path, Path(root) if root else Path.cwd(), raw, _norm_sha(path))


# ----------------------------------------------------------------------- synthetic data
SYN_DAYS = (1.3, 2.3, 6.3, 13.3, 27.3, 55.3, 90.3, 181.3, 272.3, 363.3)
SYN_FORWARD = {"BTC": 60000.0, "ETH": 3000.0}


def _month_t0(month: str, t0: str) -> int:
    y, m = (int(x) for x in month.split("-"))
    hh, mm, ss = (int(x) for x in t0.split(":"))
    return int(datetime(y, m, 1, hh, mm, ss, tzinfo=UTC).timestamp()) * 1_000_000


SYN_GENERATOR_KEYS = (
    "truth",
    "truth_arbitrage_free",
    "noise",
    "spread",
    "spread_wing",
    "n_strikes",
    "t0",
    "horizon_min",
)


def synthetic_generator(cfg: StudyConfig, underlyings: tuple[str, ...]) -> dict[str, Any]:
    """Every configurable input of the synthetic snapshot generator."""
    d = cfg.raw["data"]
    return {
        "seed": cfg.seed,
        "underlyings": list(underlyings),
        **{k: d.get(k) for k in SYN_GENERATOR_KEYS},
    }


def ensure_synthetic(cfg: StudyConfig, underlyings: tuple[str, ...]) -> None:
    """Generate synthetic months that are missing or were made with other generator inputs.

    Deterministic in the study seed and month. A month whose ``meta.json`` records other
    generator inputs (or none) is regenerated; finished units on it are then caught by the
    snapshot hash check in :func:`run_study`.
    """
    d = cfg.raw["data"]
    truths = tuple(d.get("truth", ["essvi"]))
    generator = json.loads(json.dumps(synthetic_generator(cfg, underlyings)))
    for i, month in enumerate(sorted(d["months"])):
        out = cfg.cache_dir / f"{month}-01"
        meta_p = out / "meta.json"
        if meta_p.exists():
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            if old.get("generator") == generator:
                continue
        t0 = _month_t0(month, str(d.get("t0", "00:01:00")))
        t1 = t0 + int(round(float(d["horizon_min"]) * 60e6))
        seed = int(hashlib.sha256(f"{cfg.seed}:{month}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        family = truths[i % len(truths)]
        snaps0, snaps1 = [], []
        for und in underlyings:
            exps = tuple(t0 - 60_000_000 + int(dd * 86400e6) for dd in SYN_DAYS)
            T0 = tuple((e - t0) / 1e6 / YEAR_SECONDS for e in exps)
            T1 = tuple((e - t1) / 1e6 / YEAR_SECONDS for e in exps)
            surf = synthetic.random_surface(
                family, rng, T0, arbitrage_free=bool(d.get("truth_arbitrage_free", True))
            )
            F0 = SYN_FORWARD.get(und, 100.0) * float(rng.uniform(0.7, 1.4))
            fwd0 = tuple(F0 * (1.0 + 0.03 * t) for t in T0)
            move = float(rng.normal(0.0, 0.003))
            fwd1 = tuple(f * (1.0 + move) for f in fwd0)
            surf1 = synthetic.ScaledSurface(
                T1,
                surf.status,
                {},
                base=replace(surf, expiries=T1),
                factor=1.0 + float(rng.normal(0.0, 0.01)),
            )
            kw = {
                "underlying": und,
                "n_strikes": int(d.get("n_strikes", 25)),
                "noise": float(d.get("noise", 0.005)),
                "spread": float(d.get("spread", 0.02)),
                "spread_wing": float(d.get("spread_wing", 0.5)),
            }
            sub = int(rng.integers(0, 2**31))
            s0 = synthetic.make_quote_snapshot(
                surf, forward=fwd0, time_us=t0, expirations_us=exps, seed=sub, **kw
            )
            # t1 lists the same strikes (a fixed exchange grid) at the moved forwards.
            grid = tuple(np.unique(s0.strike[s0.expiration_us == e]) for e in exps)
            s1 = synthetic.make_quote_snapshot(
                surf1,
                forward=fwd1,
                time_us=t1,
                expirations_us=exps,
                seed=sub + 1,
                strikes=grid,
                **kw,
            )
            snaps0.append(s0)
            snaps1.append(s1)
        meta = {
            "month": month,
            "source": "synthetic",
            "truth": family,
            "generator": generator,
            "status": "ok",
            "t0_us": t0,
            "t1_us": t1,
            "package_version": __version__,
            "sha256": {
                "snap_t0.csv.gz": write_snapshot(out / "snap_t0.csv.gz", merge_snapshots(snaps0)),
                "snap_t1.csv.gz": write_snapshot(out / "snap_t1.csv.gz", merge_snapshots(snaps1)),
            },
        }
        atomic_write_json(out / "meta.json", meta)


# ------------------------------------------------------------------------ discovery
def discover_units(cfg: StudyConfig) -> tuple[list[tuple[str, dict[str, Any]]], list[dict]]:
    """Units ready to run and months that are missing (with a reason)."""
    filt = FilterConfig.load(cfg.filter_path)
    if cfg.source == "synthetic":
        ensure_synthetic(cfg, filt.underlyings)
    if cfg.source in ("tardis", "synthetic"):
        labels = [f"{m}-01" for m in sorted(cfg.raw["data"]["months"])]
    elif cfg.source == "deribit":
        labels = (
            sorted(p.name for p in cfg.cache_dir.iterdir() if p.is_dir())
            if cfg.cache_dir.exists()
            else []
        )
        labels = [x for x in labels if x != "raw"]
    else:
        raise ValueError(f"unknown data source {cfg.source!r}")
    fit = cfg.raw["fit"]
    units: list[tuple[str, dict[str, Any]]] = []
    missing: list[dict[str, Any]] = []
    for label in labels:
        d = cfg.cache_dir / label
        meta_p = d / "meta.json"
        if not meta_p.exists():
            missing.append({"date": label, "reason": "not downloaded"})
            continue
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("status") != "ok" or not (d / "snap_t1.csv.gz").exists():
            missing.append({"date": label, "reason": str(meta.get("status", "unknown"))})
            continue
        for und in filt.underlyings:
            uid = f"{label}_{und}"
            units.append(
                (
                    uid,
                    {
                        "unit_id": uid,
                        "date": label,
                        "underlying": und,
                        "snap_t0": str(d / "snap_t0.csv.gz"),
                        "snap_t1": str(d / "snap_t1.csv.gz"),
                        "sha256": meta.get("sha256", {}),
                        "filter": str(cfg.filter_path),
                        "models": list(fit["models"]),
                        "sabr_beta": float(fit.get("sabr_beta", 1.0)),
                        "n_folds": int(fit.get("n_folds", 5)),
                        "floor": float(fit.get("weight_floor_iv", 0.005)),
                        "seed": cfg.seed,
                        "arbitrage": cfg.raw.get("arbitrage", {}),
                        "truth": meta.get("truth"),
                    },
                )
            )
    return units, missing


# ------------------------------------------------------------------------- one unit
def _atm30(slices: list[Slice]) -> float:
    """30-day ATM IV from the stage-1 ``theta*`` values, linear in ``T`` (flat outside)."""
    T = np.array([s.T for s in slices])
    th = np.array([atm_total_variance(s.k, s.mid_iv**2 * s.T) for s in slices])
    t30 = 30.0 / 365.0
    theta = float(np.interp(t30, T, th)) if T.size > 1 else float(th[0] / T[0] * t30)
    if t30 < T[0]:
        theta = float(th[0] / T[0] * t30)
    elif t30 > T[-1]:
        theta = float(th[-1] / T[-1] * t30)
    return math.sqrt(max(theta, 0.0) / t30)


def _nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {k: _nan_to_none(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, np.generic):
        return _nan_to_none(x.item())
    return x


def run_unit(task: dict[str, Any]) -> dict[str, Any]:
    """Fit all models to one unit and compute every metric (DESIGN 6.2, 6.3)."""
    p0, p1 = Path(task["snap_t0"]), Path(task["snap_t1"])
    shas = {"snap_t0.csv.gz": file_sha256(p0), "snap_t1.csv.gz": file_sha256(p1)}
    for name, sha in shas.items():
        expected = task.get("sha256", {}).get(name)
        if expected and expected != sha:
            raise RuntimeError(f"{name}: SHA-256 differs from the download metadata")
    filt = FilterConfig.load(Path(task["filter"]))
    und = task["underlying"]
    snap0, snap1 = read_snapshot(p0), read_snapshot(p1)
    f0, f1 = apply_filter(snap0, und, filt), apply_filter(snap1, und, filt)
    out: dict[str, Any] = {
        "date": task["date"],
        "underlying": und,
        "truth": task.get("truth"),
        "snapshot_sha256": shas,
        "counts_t0": f0.counts,
        "counts_t1": f1.counts,
        "dropped_expiries_t0": f0.dropped_expiries,
        "n_slices": len(f0.slices),
        "n_quotes": f0.n_quotes,
    }
    if len(f0.slices) == 0:
        out["empty"] = True
        return _nan_to_none(out)
    out["empty"] = False
    out["data_quality"] = data_quality(snap0, f0)
    out["atm30_iv"] = _atm30(f0.slices)
    syms0 = {str(s) for sl in f0.slices for s in sl.symbol}
    syms1 = {str(s) for sl in f1.slices for s in sl.symbol}
    out["present_at_t1_share"] = len(syms0 & syms1) / len(syms0) if syms0 else None

    slices = f0.slices
    n_folds, floor, seed = task["n_folds"], task["floor"], task["seed"]
    offsets = fold_offsets(seed, task["unit_id"], len(slices), n_folds)
    folds = [fold_ids(s.n, int(offsets[j]), n_folds) for j, s in enumerate(slices)]
    full_in = to_fit_input(slices, seed=seed, floor=floor)
    wing_in = to_fit_input(slices, seed=seed, floor=floor, masks=[wing_mask(s.n) for s in slices])
    fold_in = [
        to_fit_input(slices, seed=seed, floor=floor, masks=[fd != f for fd in folds])
        for f in range(n_folds)
    ]
    arb_cfg = ArbitrageConfig(
        grid_z=tuple(task["arbitrage"].get("grid_z", (-4.0, 4.0))),  # type: ignore[arg-type]
        n_grid=int(task["arbitrage"].get("n_grid", 401)),
        g_tol=float(task["arbitrage"].get("g_tol", 1e-6)),
        calendar_tol=float(task["arbitrage"].get("calendar_tol", 1e-8)),
    )

    fits: dict[str, dict[str, Any]] = {}
    for name in task["models"]:
        model = get_model(name, sabr_beta=task["sabr_beta"])
        t = time.perf_counter()
        fold_fits = [model.fit(d) for d in fold_in]
        wing_fit = model.fit(wing_in)
        full_fit = model.fit(full_in)
        elapsed = time.perf_counter() - t
        all_fits = [*fold_fits, wing_fit, full_fit]
        ok = np.array([[accepted(st) for st in f.status] for f in all_fits]).all(axis=0)
        n_failed = sum(not accepted(st) for f in all_fits for st in f.status)
        fits[name] = {
            "folds": fold_fits,
            "wing": wing_fit,
            "full": full_fit,
            "ok": ok,
            "time": elapsed,
            "n_failed": n_failed,
            "status_full": list(full_fit.status),
        }
    common = np.all([fits[m]["ok"] for m in fits], axis=0)
    out["n_common_slices"] = int(common.sum())
    out["n_common_quotes"] = int(sum(s.n for s, c in zip(slices, common, strict=True) if c))

    # t1 quotes on common expiries; persistence on instruments filtered at both times.
    t1_by_exp = {s.expiration_us: s for s in f1.slices}
    pers_pred: list[float] = []
    pers_obs: list[float] = []
    pairs: list[tuple[int, Slice, np.ndarray, np.ndarray]] = []
    for j, s0 in enumerate(slices):
        s1 = t1_by_exp.get(s0.expiration_us)
        if not common[j] or s1 is None:
            continue
        idx0 = {str(sym): i for i, sym in enumerate(s0.symbol)}
        both1 = np.array([str(sym) in idx0 for sym in s1.symbol], dtype=bool)
        both0 = np.array([idx0[str(sym)] for sym in s1.symbol[both1]], dtype=int)
        pairs.append((j, s1, both1, both0))
        pers_pred.extend(s0.mid_iv[both0].tolist())
        pers_obs.extend(s1.mid_iv[both1].tolist())
    out["persistence_rmse_iv"] = rmse_volpt(pers_pred, pers_obs) if pers_obs else float("nan")
    out["n_next_quotes"] = int(sum(s1.n for _, s1, _, _ in pairs))
    out["n_next_common_quotes"] = len(pers_obs)

    models_out: dict[str, Any] = {}
    for name, fr in fits.items():
        cv_p, cv_o, cv_b, cv_a, cv_h = [], [], [], [], []
        wing_p, wing_o, fit_p, fit_o = [], [], [], []
        for j, s in enumerate(slices):
            if not common[j]:
                continue
            for f in range(n_folds):
                held = folds[j] == f
                cv_p.append(fr["folds"][f].iv(j, s.k[held]))
                cv_o.append(s.mid_iv[held])
                cv_b.append(s.bid_iv[held])
                cv_a.append(s.ask_iv[held])
                cv_h.append(s.half_spread[held])
            out_idx = np.array([0, s.n - 1])
            wing_p.append(fr["wing"].iv(j, s.k[out_idx]))
            wing_o.append(s.mid_iv[out_idx])
            fit_p.append(fr["full"].iv(j, s.k))
            fit_o.append(s.mid_iv)
        nx_p, nx_o, nc_p, nc_o = [], [], [], []
        for j, s1, both1, _ in pairs:
            pred = fr["full"].iv(j, s1.k)
            nx_p.append(pred)
            nx_o.append(s1.mid_iv)
            nc_p.append(pred[both1])
            nc_o.append(s1.mid_iv[both1])

        def cat(xs: list[np.ndarray]) -> np.ndarray:
            return np.concatenate(xs) if xs else np.array([])

        diag = diagnose(fr["full"], [s.k for s in slices], arb_cfg)
        models_out[name] = {
            "cv_rmse_iv": rmse_volpt(cat(cv_p), cat(cv_o)),
            "cv_mae_iv": mae_volpt(cat(cv_p), cat(cv_o)),
            "cv_inside_spread": inside_spread(cat(cv_p), cat(cv_b), cat(cv_a)),
            "cv_rms_spread_units": rms_spread_units(cat(cv_p), cat(cv_o), cat(cv_h), floor),
            "wing_rmse_iv": rmse_volpt(cat(wing_p), cat(wing_o)),
            "fit_rmse_iv": rmse_volpt(cat(fit_p), cat(fit_o)),
            "next_rmse_iv": rmse_volpt(cat(nx_p), cat(nx_o)),
            "next_rmse_iv_common": rmse_volpt(cat(nc_p), cat(nc_o)),
            "n_slices_checked": diag["n_slices"],
            "n_bfly_wide": diag["n_bfly_wide"],
            "n_bfly_inrange": diag["n_bfly_inrange"],
            "n_price_wide": diag["n_price_wide"],
            "n_slope": diag["n_slope"],
            "min_g_wide": diag["min_g_wide"],
            "n_pairs": diag["n_pairs"],
            "n_cal": diag["n_cal"],
            "worst_cal_gap": diag["worst_cal_gap"],
            "fit_time_s": fr["time"],
            "n_failed_slices": fr["n_failed"],
            "status_full": fr["status_full"],
        }
    out["models"] = models_out
    return _nan_to_none(out)


# ------------------------------------------------------------------------ the study
def _manifest(cfg: StudyConfig) -> dict[str, Any]:
    filt = FilterConfig.load(cfg.filter_path)
    fit = cfg.raw["fit"]
    return {
        "study": cfg.name,
        "source": cfg.source,
        "filter_sha256": filt.sha256,
        "filter_version": filt.version,
        "package_version": __version__,
        "seed": cfg.seed,
        "models": list(fit["models"]),
        "fit": {k: fit[k] for k in sorted(fit) if k != "filter"},
        "arbitrage": cfg.raw.get("arbitrage", {}),
        "horizon_min": cfg.raw["data"].get("horizon_min"),
        "t0": cfg.raw["data"].get("t0"),
        "generator": (
            synthetic_generator(cfg, filt.underlyings) if cfg.source == "synthetic" else None
        ),
    }


def expected_unit_ids(cfg: StudyConfig) -> set[str] | None:
    """Unit ids the configuration covers (``None`` for the deribit source: every pair)."""
    if cfg.source == "deribit":
        return None
    filt = FilterConfig.load(cfg.filter_path)
    return {f"{m}-01_{u}" for m in cfg.raw["data"]["months"] for u in filt.underlyings}


def check_run_matches(cfg: StudyConfig) -> None:
    """Raise :class:`RunMismatchError` unless the run directory was made by this config."""
    path = cfg.out_dir / "manifest.json"
    if not path.exists():
        raise RunMismatchError(f"no run found for study {cfg.name!r}; run `study run` first")
    stored = json.loads(path.read_text(encoding="utf-8"))
    current = json.loads(json.dumps(_manifest(cfg), sort_keys=True))
    if stored != current:
        diff = sorted(k for k in set(stored) | set(current) if stored.get(k) != current.get(k))
        raise RunMismatchError(
            "the run directory was produced with a different configuration "
            f"(changed: {', '.join(diff)}); rerun `study run --force-new` before reporting"
        )


def run_study(
    config: Path,
    *,
    workers: int | None = None,
    force_new: bool = False,
    limit: int | None = None,
    root: Path | None = None,
) -> dict[str, int]:
    """Run every pending unit; finished units are skipped, failed ones retried (up to 3)."""
    cfg = load_study(config, root)
    store = UnitStore(cfg.out_dir)
    store.open(_manifest(cfg), force_new=force_new)
    units, missing = discover_units(cfg)
    # A finished unit whose snapshot changed invalidates the run.
    for uid, task in units:
        rec = store.read(uid)
        if rec and rec.get("status") == "ok":
            old = rec.get("snapshot_sha256", {})
            new = task.get("sha256", {})
            if new and old and old != new:
                raise RunMismatchError(
                    f"snapshot of {uid} changed since it was run; use --force-new"
                )
    if limit is not None:
        pending = [(uid, task) for uid, task in units if not store.is_final(uid)]
        units = pending[: max(0, int(limit))]
    n_workers = int(workers if workers is not None else cfg.raw["study"].get("workers", 1))
    store.log("-", "start", 0.0, f"{len(units)} units, {len(missing)} dates missing")
    t = time.perf_counter()
    counts = run_units(store, units, run_unit, workers=n_workers)
    elapsed = time.perf_counter() - t
    store.log("-", "end", elapsed, json.dumps(counts))
    # A resume that runs nothing (e.g. the second pass of a retry loop) keeps the timing of
    # the last pass that did run units.
    info_p = cfg.out_dir / "run_info.json"
    prev = {}
    if info_p.exists() and not force_new:
        prev = json.loads(info_p.read_text(encoding="utf-8"))
    ran = counts.get("ok", 0) + counts.get("error", 0) > 0
    atomic_write_json(
        info_p,
        {
            "missing": missing,
            "n_units": len(units),
            "last_run_wall_s": elapsed if ran else prev.get("last_run_wall_s", elapsed),
            "total_run_wall_s": float(prev.get("total_run_wall_s", 0.0))
            + (elapsed if ran else 0.0),
            "workers": min(max(n_workers, 1), 3)
            if ran
            else prev.get("workers", min(max(n_workers, 1), 3)),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "package_version": __version__,
        },
    )
    return counts


def fit_table(snapshot: Path, filter_path: Path, underlying: str) -> str:
    """Fit every model to one snapshot (all quotes) and return a plain-text table."""
    filt = FilterConfig.load(filter_path)
    snap = read_snapshot(snapshot)
    f = apply_filter(snap, underlying, filt)
    if not f.slices:
        return f"{underlying}: no slices pass the filter ({f.counts})"
    data = to_fit_input(f.slices)
    lines = [
        f"{underlying}: {len(f.slices)} slices, {f.n_quotes} quotes; filter counts {f.counts}",
        f"{'model':<9} {'fit RMSE':>9} {'bfly':>6} {'cal':>6} {'failed':>7} {'time':>7}",
    ]
    for name in ("baseline", "svi", "ssvi", "essvi", "sabr"):
        t = time.perf_counter()
        fit = get_model(name).fit(data)
        el = time.perf_counter() - t
        pred = np.concatenate([fit.iv(j, data.k[j]) for j in range(data.n_slices)])
        obs = np.concatenate(data.iv_mid)
        ok = np.isfinite(pred)
        dg = diagnose(fit, list(data.k))
        lines.append(
            f"{name:<9} {rmse_volpt(pred[ok], obs[ok]):>8.3f}v "
            f"{dg['n_bfly_wide']:>3}/{dg['n_slices']:<2} {dg['n_cal']:>3}/{dg['n_pairs']:<2} "
            f"{sum(not accepted(s) for s in fit.status):>7} {el:>6.2f}s"
        )
    lines.append("(fit RMSE in vol points on all filtered quotes; bfly = wide-grid butterfly)")
    return "\n".join(lines)
