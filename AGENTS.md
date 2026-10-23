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
| `configs/filter_v1.toml` | pre-registered quote filter (provisional until M4) | M4 |
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

## How to run (after M4-M5)

```sh
uv run surfbench data tardis --config configs/study_quick.toml    # resumable; stops on quota
uv run surfbench data status --config configs/study_quick.toml
for i in 1 2 3; do uv run surfbench study run --config configs/study_quick.toml --workers 3 && break; done
uv run surfbench study report --config configs/study_quick.toml
```

Full size: the same with `configs/study_full.toml` (91 months, 60-minute horizon; compute
projected 6-15 minutes, download projected 15-30 GB over many quota windows). Watch
`runs/<name>/progress.log` for stalls.

## Status and next steps

Updated: 2026-10-24.

Done:
- Scaffold: package skeleton, CLI stub, configs, CI, hygiene files, `docs/DESIGN.md`.
- Research checks recorded in DESIGN (Tardis terms and file layout, Deribit coin-price
  convention, reference values agreeing with QuantLib 1.44).
- M1: `black` (prices, vega, delta, safeguarded Newton-bisection IV inversion), `stats`
  (bootstrap and moving-block bootstrap, pooled-ratio bootstrap, sign-flip test, Holm, rank
  probabilities), `checkpoint` (atomic unit files, manifest check, progress log, path
  scrubbing, process pool capped at 3, retries up to 3 attempts). Tests in
  `tests/test_black.py`, `test_stats.py`, `test_checkpoint.py`.
- M2: the five models (`models/`: baseline, raw SVI quasi-explicit + polish, SSVI power
  law, eSSVI Corbetta et al. with a `rho` polish, SABR Hagan) behind one objective, and
  `synthetic` (random admissible surfaces, noisy snapshots). Noise-free own-family recovery
  is below 1e-7 vol point for every model; constraints are asserted on every accepted
  fit. Deviations in DESIGN section 13. Optional svi-py/QuantLib oracle fixtures were not
  produced (SABR is already checked against QuantLib through the reference values).
- M3: `arbitrage` (Durrleman `g` on wide and in-range grids, call-price monotonicity and
  convexity cross-check, calendar gaps between adjacent accepted slices, slope bound).
  Vogt slice, admissible SSVI sets, eSSVI butterfly/calendar examples match the reference
  values; `g` and price checks agree on random SVI slices; noisy SSVI/eSSVI fits show no
  violations.

Blocked or watch:
- The Tardis free download returned `429` with `Retry-After` about 24 hours on 2026-10-23
  around 21:45 UTC. Do not request Tardis files before 2026-10-24 22:00 UTC.

Next (in order): M4 filter freeze on live Deribit
snapshots, then Tardis extraction as the quota allows; M5 runner and report; M6 quick study.
