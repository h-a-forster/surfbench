"""Repository hygiene: committed results hold aggregate statistics only, no local paths."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_KEYS = {
    "bid_price",
    "ask_price",
    "mark_price",
    "mid_iv",
    "bid_iv",
    "ask_iv",
    "iv_mid",
    "params",
    "slices",
    "theta",
    "theta_star",
    "atm30_iv",
    "underlying_price",
    "strike",
}
PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/home/[a-z]"),
    re.compile(r"/Users/[A-Za-z]"),
]


def _keys(obj: object) -> set[str]:
    if isinstance(obj, dict):
        out = set(obj)
        for v in obj.values():
            out |= _keys(v)
        return out
    if isinstance(obj, list):
        out: set[str] = set()
        for v in obj:
            out |= _keys(v)
        return out
    return set()


def test_results_contain_no_quote_level_fields() -> None:
    for p in (ROOT / "results").rglob("summary.json"):
        keys = _keys(json.loads(p.read_text(encoding="utf-8")))
        assert not keys & FORBIDDEN_KEYS, (p.name, keys & FORBIDDEN_KEYS)


def test_no_market_data_files_in_results() -> None:
    bad = [p for p in (ROOT / "results").rglob("*") if p.suffix in {".gz", ".csv", ".parquet"}]
    assert bad == []


def test_no_local_paths_in_committed_text() -> None:
    files = [
        *ROOT.glob("*.md"),
        *ROOT.glob("*.toml"),
        *(ROOT / "docs").rglob("*.md"),
        *(ROOT / "configs").rglob("*.toml"),
        *(ROOT / "results").rglob("*.*"),
    ]
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for pat in PATH_PATTERNS:
            assert not pat.search(text), (p.name, pat.pattern)


def test_every_package_module_is_tracked_by_git() -> None:
    """A broad ignore rule once hid src/surfbench/data/ from git and from the wheel."""
    import shutil
    import subprocess

    import pytest

    git = shutil.which("git")
    if git is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    pkg = ROOT / "src" / "surfbench"
    modules = sorted(p.relative_to(ROOT).as_posix() for p in pkg.rglob("*.py"))
    ignored = subprocess.run(
        [git, "check-ignore", "--no-index", *modules],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    assert ignored == [], f"package modules matched by .gitignore: {ignored}"
