# AGENTS.md

Guide for coding agents (and humans) working on surfbench. Read `docs/DESIGN.md` before
changing any model, diagnostic or statistic: it holds the formulas, constraints, reference
values, study design and milestones this file points to.

## What this is

A benchmark of implied volatility surface fits (raw SVI, SSVI, eSSVI, SABR, quadratic
baseline) on Deribit BTC and ETH options: cross-validated accuracy, next-snapshot error and
static-arbitrage violations, with month-level bootstrap intervals and Holm-adjusted
comparisons.

## Conventions

- uv + hatchling, `src/` layout, Python >= 3.11 (local dev pinned to 3.13 by
  `.python-version`), typed code, `ruff check` + `ruff format` (line length 100), pytest.
  Runtime dependencies: numpy and scipy only. Ask before adding one.
- Units: `k = ln(K/F)`, `T` in years ACT/365 on a 24/7 clock, total variance
  `w = sigma^2 T`, volatilities as decimals (reports show vol points), zero rates, coin
  prices = Black-76 / F (DESIGN section 4).
- All models share one objective (DESIGN 4.2). Constraints in DESIGN 4.3 are asserted on
  every accepted fit; optimiser failures are reported, never silently replaced.
- Every stochastic function takes `seed`. Randomisation p-values use `(1 + count)/(B + 1)`.
- No LLM or model API calls. CPU only. At most 3 worker processes, BLAS threads set to 1.
- Long jobs: every finished unit goes through `surfbench.checkpoint.UnitStore` (atomic file
  per unit, skip on rerun, `progress.log`). Never keep results only in memory.
- **Data licence** (DESIGN section 3): never commit quotes, snapshots, fitted parameters or
  IV levels. `cache/`, `data/`, `runs/` and `*.csv.gz` are gitignored. Commit only download
  metadata and aggregate statistics under `results/`. Tests use synthetic data only and never
  touch the network.
- **Tardis downloads**: plain `GET` only (HEAD gives 404, Range gives 403), one request at a
  time, honour `429 Retry-After` by stopping the session (`cache/tardis/_quota.json`).
  Do not retry a locked-out quota.
- Scratch files go in `tmp/` (gitignored). No `.bak` copies; use git.
- No absolute local paths, user names or machine details in committed files or logs.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Do not push and
  do not create remote repositories; publishing is done separately after review.
- Statistical honesty: report sizes, seeds, versions and intervals; Holm on the primary
  family only (DESIGN 6.1); never overstate. Before calling anything a third-party bug,
  produce a minimal repro and rule out an adapter or convention mistake (DESIGN 7.5).

## Architecture map

See DESIGN section 8 for the data flow. Every module under `src/surfbench/` currently holds a
docstring naming its DESIGN section and milestone.

| path | role | milestone |
|---|---|---|
| `black.py`, `stats.py`, `checkpoint.py` | numerics, inference, resumable units | M1 |
| `models/`, `synthetic.py` | five models, synthetic snapshots | M2 |
| `arbitrage.py` | butterfly, price convexity, calendar, slopes | M3 |
| `quotes.py`, `data/` | filter, forwards, Tardis extractor, Deribit live adapter | M4 |
| `metrics.py`, `study/`, `cli.py` | CV, next-snapshot, runner, report | M5 |
| `configs/filter_v1.toml` | pre-registered quote filter (frozen 2026-10-24) | M4 |
| `configs/study_quick.toml`, `study_full.toml` | study sizes | M5 |
| `scripts/reference_values.py` | independent oracle values (DESIGN 7.1) | done |

## How to test

```sh
uv sync --group dev
uv run pytest                 # fast suite (CI)
uv run pytest --runslow       # plus slow checks
uv run ruff check .
uv run ruff format --check .
uv run python scripts/reference_values.py
```

## How to run

```sh
uv run surfbench data tardis --config configs/study_quick.toml    # resumable; stops on quota
uv run surfbench data status --config configs/study_quick.toml
for i in 1 2 3; do uv run surfbench study run --config configs/study_quick.toml --workers 3 && break; done
uv run surfbench study report --config configs/study_quick.toml
```

Full size: the same with `configs/study_full.toml` (91 months, 60-minute horizon; compute
about 30 minutes, download projected 15-30 GB over many quota windows). Watch
`runs/<name>/progress.log` for stalls.

## Status and next steps

Updated: 2026-10-30.

Done (v0.1.0 in CHANGELOG, dated 2026-10-30; not tagged yet):
- M1-M5 complete: numerics, statistics and checkpointing; the five models behind one
  objective; arbitrage diagnostics; quote filter, Tardis extractor and Deribit adapter
  (tested offline); metrics, study runner, report and CLI. Deviations in DESIGN section 13.
- M6 synthetic part: `configs/study_synthetic.toml`, 24 dates x 2 underlyings from
  arbitrage-free eSSVI/SVI/SABR truths, results in `results/synthetic-v1/` (last run 134 s
  on 3 workers), written up in REPORT.md sections 1-2.
- Calibrator check against scipy differential evolution (`results/calibrator-check/`).
- Filter v1 frozen on 2026-10-24 after 16 live snapshots (REPORT.md section 4).
- Review fixes of 2026-10-24 (anchored ignore rules, eSSVI sufficient calendar conditions,
  run provenance and resume, Tardis retries): CHANGELOG 0.1.0 "Fixed", DESIGN 13 items 14-15.
- Docs and release notes (`docs/release-notes-0.1.0.md`) state that every published result
  is synthetic.

Open:
- **Market-data quick study (M6 proper) not run.** No market-data result exists. It needs
  the Tardis free downloads (projected 2-3 GB streamed, 12 files per session, a 24-hour
  lockout after about a dozen files was observed on 2026-10-23) and about 9 minutes of
  compute. Steps:
  1. `uv run surfbench data tardis --config configs/study_quick.toml`, repeated on later
     days until `data status` shows every month done or unavailable; stop on 429.
  2. `for i in 1 2 3; do uv run surfbench study run --config configs/study_quick.toml --workers 3 && break; done`,
     then `uv run surfbench study report --config configs/study_quick.toml`; commit
     `results/quick-v1/`; rewrite REPORT.md sections 2-3 and 5 and the README status with
     the real numbers (descriptive only if fewer than 12 dates are usable).
- CI has never run on GitHub (the workflow now also triggers on pushes to `main`). Check
  the first run; actions are pinned by tag, not commit SHA.
- Release: tag `v0.1.0` and publish the GitHub release from `docs/release-notes-0.1.0.md`
  (owner).
- Optional: svi-py / QuantLib oracle fixtures (DESIGN 7.5); eSSVI and SSVI in the calibrator
  check; full study.
