"""Scaffold checks: the package imports, exposes its version, and the CLI starts."""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

import surfbench
from surfbench.cli import main

ROOT = Path(__file__).resolve().parents[1]

MODULES = [
    "surfbench.black",
    "surfbench.stats",
    "surfbench.checkpoint",
    "surfbench.arbitrage",
    "surfbench.metrics",
    "surfbench.quotes",
    "surfbench.synthetic",
    "surfbench.models",
    "surfbench.models.base",
    "surfbench.models.svi",
    "surfbench.models.ssvi",
    "surfbench.models.essvi",
    "surfbench.models.sabr",
    "surfbench.models.baseline",
    "surfbench.data",
    "surfbench.data.tardis",
    "surfbench.data.deribit",
    "surfbench.study",
    "surfbench.study.runner",
    "surfbench.study.report",
]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import(name: str) -> None:
    importlib.import_module(name)


def test_version_is_a_string() -> None:
    assert isinstance(surfbench.__version__, str) and surfbench.__version__


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("surfbench ")


def test_cli_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 0
    assert "surfbench" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["filter_v1.toml", "study_quick.toml", "study_full.toml"])
def test_configs_parse(name: str) -> None:
    with (ROOT / "configs" / name).open("rb") as fh:
        cfg = tomllib.load(fh)
    assert cfg


def test_quick_months_are_first_of_month() -> None:
    with (ROOT / "configs" / "study_quick.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    months = cfg["data"]["months"]
    assert len(months) == len(set(months)) >= 12
    assert all(len(m) == 7 and m[4] == "-" for m in months)
