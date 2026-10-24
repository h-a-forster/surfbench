"""Resumable unit store (DESIGN 7.3)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from surfbench import checkpoint
from surfbench.checkpoint import RunMismatchError, UnitStore, run_units

CALLS: list[str] = []


def _square(x: int) -> dict[str, int]:
    CALLS.append(str(x))
    return {"value": x * x}


def _fail_on_odd(x: int) -> dict[str, int]:
    if x % 2:
        raise RuntimeError("bad unit at C:\\Users\\someone\\data\\file.csv and /home/someone/x.csv")
    return {"value": x}


def test_resume_skips_finished(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({"config": "a"})
    CALLS.clear()
    units = [(f"u{i}", i) for i in range(4)]
    assert run_units(store, units, _square) == {"ok": 4, "error": 0, "skipped": 0}
    assert CALLS == ["0", "1", "2", "3"]
    CALLS.clear()
    assert run_units(store, units, _square) == {"ok": 0, "error": 0, "skipped": 4}
    assert CALLS == []
    assert store.read("u3")["value"] == 9
    log = (tmp_path / "run" / "progress.log").read_text(encoding="utf-8").splitlines()
    assert len(log) == 4 and all("\tok\t" in line for line in log)


def test_errors_retried_up_to_max(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({"config": "a"})
    units = [(f"u{i}", i) for i in range(3)]
    for attempt in range(1, checkpoint.MAX_ATTEMPTS + 2):
        counts = run_units(store, units, _fail_on_odd)
        if attempt <= checkpoint.MAX_ATTEMPTS:
            assert counts["error"] == 1
        else:
            assert counts == {"ok": 0, "error": 0, "skipped": 3}
    rec = store.read("u1")
    assert rec["status"] == "error" and rec["attempts"] == checkpoint.MAX_ATTEMPTS
    assert "someone" not in rec["error"] and "<path>/file.csv" in rec["error"]
    assert "someone" not in (tmp_path / "run" / "progress.log").read_text(encoding="utf-8")


def test_manifest_mismatch(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({"config": "a", "filter": "f"})
    run_units(store, [("u0", 0)], _square)
    with pytest.raises(RunMismatchError, match="config"):
        UnitStore(tmp_path / "run").open({"config": "b", "filter": "f"})
    store2 = UnitStore(tmp_path / "run")
    store2.open({"config": "b", "filter": "f"}, force_new=True)
    assert store2.read("u0") is None
    UnitStore(tmp_path / "run").open({"config": "b", "filter": "f"})  # unchanged: resumes


def test_parallel_pool(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({})
    counts = run_units(store, [(f"u{i}", i) for i in range(6)], _fail_on_odd, workers=8)
    assert counts == {"ok": 3, "error": 3, "skipped": 0}
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    assert len(store.load_all()) == 6


def test_atomic_write_and_invalid_id(tmp_path: Path) -> None:
    p = tmp_path / "a" / "b.json"
    checkpoint.atomic_write_json(p, {"x": 1})
    assert json.loads(p.read_text()) == {"x": 1}
    assert not list(p.parent.glob("*.tmp"))
    with pytest.raises(ValueError):
        UnitStore(tmp_path).path("../evil")


def test_corrupt_unit_file_is_rerun(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({})
    store.path("u0").write_text("{not json", encoding="utf-8")
    assert run_units(store, [("u0", 2)], _square)["ok"] == 1


def test_scrub_paths() -> None:
    s = checkpoint.scrub_paths("open C:\\Users\\me\\proj\\x.csv.gz failed; /home/me/a/b.txt")
    assert s == "open <path>/x.csv.gz failed; <path>/b.txt"
    assert checkpoint.scrub_paths("ratio 1/2 and 3/4") == "ratio 1/2 and 3/4"


def test_force_new_resets_even_when_manifest_is_unchanged(tmp_path: Path) -> None:
    store = UnitStore(tmp_path / "run")
    store.open({"config": "a"})
    run_units(store, [("u0", 0)], _square)
    assert store.read("u0") is not None
    UnitStore(tmp_path / "run").open({"config": "a"}, force_new=True)
    assert UnitStore(tmp_path / "run").read("u0") is None
    assert "--force-new" in (tmp_path / "run" / "progress.log").read_text(encoding="utf-8")
