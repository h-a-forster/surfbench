"""Metrics, runner, report and CLI on synthetic data (DESIGN 6, 7.2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from surfbench import metrics
from surfbench.cli import main
from surfbench.study.report import build_summary, write_report
from surfbench.study.runner import load_study, run_study

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, months: list[str], models: list[str], truth: str = "essvi") -> Path:
    filt = (ROOT / "configs" / "filter_v1.toml").as_posix()
    text = f"""
[study]
name = "test"
out_dir = "{(tmp_path / "run").as_posix()}"
results_dir = "{(tmp_path / "results").as_posix()}"
seed = 7
workers = 1

[data]
source = "synthetic"
cache_dir = "{(tmp_path / "cache").as_posix()}"
months = {json.dumps(months)}
horizon_min = 15
truth = ["{truth}"]
n_strikes = 15

[fit]
filter = "{filt}"
models = {json.dumps(models)}
n_folds = 5

[report]
n_boot = 500
n_perm = 2000
"""
    p = tmp_path / "study.toml"
    p.write_text(text, encoding="utf-8")
    return p


# ------------------------------------------------------------------------ metrics
def test_fold_ids_and_offsets() -> None:
    assert list(metrics.fold_ids(7, 2, 5)) == [2, 3, 4, 0, 1, 2, 3]
    a = metrics.fold_offsets(1, "2024-01-01_BTC", 4)
    assert np.array_equal(a, metrics.fold_offsets(1, "2024-01-01_BTC", 4))
    assert not np.array_equal(
        np.concatenate([metrics.fold_offsets(1, f"u{i}", 4) for i in range(5)]),
        np.zeros(20),
    )
    with pytest.raises(ValueError):
        metrics.fold_ids(5, 0, 1)


def test_metric_hand_values() -> None:
    assert metrics.rmse_volpt([0.51, 0.49], [0.50, 0.50]) == pytest.approx(1.0)
    assert metrics.mae_volpt([0.52, 0.49], [0.50, 0.50]) == pytest.approx(1.5)
    assert metrics.inside_spread([0.5, 0.6], [0.49, 0.49], [0.51, 0.51]) == 0.5
    assert metrics.rms_spread_units([0.51], [0.50], [0.002], 0.005) == pytest.approx(2.0)
    assert np.isnan(metrics.rmse_volpt([], []))
    assert list(metrics.wing_mask(4)) == [False, True, True, False]
    with pytest.raises(ValueError):
        metrics.rmse_volpt([1.0], [1.0, 2.0])


# ---------------------------------------------------------------- end to end (fast)
def test_synthetic_end_to_end_resume_and_report(tmp_path: Path) -> None:
    cfg_path = _config(tmp_path, ["2024-01", "2024-02"], ["baseline", "ssvi"])
    c1 = run_study(cfg_path, limit=1)
    assert c1 == {"ok": 1, "error": 0, "skipped": 0}
    c2 = run_study(cfg_path)
    assert c2 == {"ok": 3, "error": 0, "skipped": 1}
    c3 = run_study(cfg_path)
    assert c3 == {"ok": 0, "error": 0, "skipped": 4}
    out = write_report(cfg_path)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    board = (out / "leaderboard.md").read_text(encoding="utf-8")
    assert "| ssvi |" in board and "Holm" in board
    # Report numbers equal a direct recomputation from the unit files.
    units = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((tmp_path / "run" / "units").glob("*_BTC.json"))
    ]
    direct = np.mean([u["models"]["ssvi"]["cv_rmse_iv"] for u in units])
    got = summary["by_underlying"]["BTC"]["metrics"]["cv_rmse_iv"]["ssvi"]["estimate"]
    assert got == pytest.approx(direct, rel=1e-12)
    diff = np.mean(
        [u["models"]["baseline"]["cv_rmse_iv"] - u["models"]["ssvi"]["cv_rmse_iv"] for u in units]
    )
    t = next(
        x for x in summary["pairwise_primary"] if x["underlying"] == "BTC" and x["a"] == "baseline"
    )
    assert t["mean_diff"] == pytest.approx(diff, rel=1e-12)
    viol = sum(u["models"]["ssvi"]["n_bfly_wide"] for u in units)
    assert (
        summary["by_underlying"]["BTC"]["arbitrage"]["bfly_viol_wide"]["ssvi"]["violations"] == viol
    )
    assert viol == 0  # SSVI is arbitrage-free by construction
    assert len(summary["pairwise_primary"]) == 2 and "p_holm" in summary["pairwise_primary"][0]
    # Nothing absolute or quote-level in the committed artefacts.
    text = (out / "summary.json").read_text(encoding="utf-8") + board
    assert str(tmp_path) not in text and tmp_path.as_posix() not in text
    assert "atm30" not in text and "mid_iv" not in text


def test_changed_config_refuses_resume(tmp_path: Path) -> None:
    cfg_path = _config(tmp_path, ["2024-01"], ["baseline"])
    run_study(cfg_path)
    _config(tmp_path, ["2024-01"], ["baseline", "ssvi"])
    from surfbench.checkpoint import RunMismatchError

    with pytest.raises(RunMismatchError):
        run_study(cfg_path)
    assert main(["study", "run", "--config", str(cfg_path)]) == 2  # clean message, no traceback
    assert run_study(cfg_path, force_new=True)["ok"] == 2


def test_cli_fit_and_study(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg_path = _config(tmp_path, ["2024-03"], ["baseline"])
    assert main(["study", "run", "--config", str(cfg_path)]) == 0
    assert main(["study", "report", "--config", str(cfg_path)]) == 0
    snap = tmp_path / "cache" / "2024-03-01" / "snap_t0.csv.gz"
    assert main(["fit", str(snap), "--filter", str(ROOT / "configs" / "filter_v1.toml")]) == 0
    out = capsys.readouterr().out
    assert "units: ok 2" in out and "essvi" in out and "fit RMSE" in out
    assert main(["data", "status", "--config", str(cfg_path)]) == 0


def test_load_study_resolves_relative_paths(tmp_path: Path) -> None:
    cfg = load_study(ROOT / "configs" / "study_quick.toml", root=tmp_path)
    assert cfg.out_dir == tmp_path / "runs" / "quick-v1"
    assert cfg.source == "tardis" and len(cfg.sha256) == 64


def test_missing_tardis_months_are_reported(tmp_path: Path) -> None:
    import shutil

    from surfbench.study.runner import discover_units

    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    cfg = load_study(ROOT / "configs" / "study_quick.toml", root=tmp_path)
    units, missing = discover_units(cfg)
    assert units == [] and len(missing) == 28
    assert {m["reason"] for m in missing} == {"not downloaded"}


# ----------------------------------------------------------------- end to end (slow)
@pytest.mark.slow
def test_six_month_synthetic_study_all_models(tmp_path: Path) -> None:
    months = [f"2024-{m:02d}" for m in range(1, 7)]
    models = ["baseline", "svi", "ssvi", "essvi", "sabr"]
    cfg_path = _config(tmp_path, months, models)
    counts = run_study(cfg_path, workers=3)
    assert counts["ok"] == 12 and counts["error"] == 0
    summary = build_summary(load_study(cfg_path))
    for und in ("BTC", "ETH"):
        arb = summary["by_underlying"][und]["arbitrage"]
        for m in ("ssvi", "essvi"):
            assert arb["bfly_viol_wide"][m]["violations"] == 0
            assert arb["cal_viol"][m]["violations"] == 0
            assert arb["price_viol_wide"][m]["violations"] == 0
    assert len(summary["pairwise_primary"]) == 20
