# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-24

First version with the full pipeline. The market-data quick study has not been run yet: the
Tardis.dev free download quota was locked during development (see REPORT.md).

### Added

- `black`: normalised Black-76 prices, vega, forward delta and a vectorised safeguarded
  Newton-bisection implied-volatility inversion for coin-settled options.
- `stats`: percentile and moving-block bootstrap intervals, pooled-ratio (cluster)
  bootstrap, two-sided sign-flip test (random and exact), Holm adjustment, rank
  probabilities.
- `checkpoint`: atomic per-unit JSON files, run manifest, progress log with path scrubbing,
  process pool capped at 3 workers, error units retried up to 3 attempts.
- Models behind one objective (half-spread-weighted IV least squares): quadratic baseline,
  raw SVI (quasi-explicit calibration with a least-squares polish, Lee's bound imposed),
  SSVI with power-law curvature (global), eSSVI (Corbetta et al. robust slice-by-slice
  calibration under the Hendriks-Martini calendar conditions), SABR (Hagan lognormal,
  `beta = 1`).
- `arbitrage`: Durrleman's `g` on wide and in-range grids, call-price monotonicity and
  convexity cross-check, calendar gaps between adjacent expiries, slope bound.
- `quotes`: canonical snapshot files with SHA-256, the pre-registered quote filter with
  per-step counts, data-quality diagnostics.
- `data.tardis`: streaming extractor for the free first-of-month Tardis.dev files (early
  stop after `t1`, quota lock on HTTP 429, 404 handling, retries, cache cap, seeded month
  order); `data.deribit`: live snapshot adapter for development.
- `metrics`, `study.runner`, `study.report`: 5-fold cross-validation, wing and
  next-snapshot errors, quote persistence, arbitrage shares, bootstrap intervals, Holm over
  the 20 pairwise comparisons, exploratory calm/stress split.
- CLI: `surfbench data tardis|deribit|status`, `surfbench fit`, `surfbench study run|report`.
- `configs/study_synthetic.toml` and its committed results (`results/synthetic-v1/`).

### Notes

- `configs/filter_v1.toml` frozen on 2026-10-24 with the draft thresholds unchanged, after a
  review on 16 live Deribit snapshots (8 pairs, 22:09 to 01:54 UTC) and before any Tardis
  study month was seen; see REPORT.md section 4. Only its status comment changed, so its
  SHA-256 changed and the synthetic study was rerun with the frozen file.
- Deviations from the design are listed in `docs/DESIGN.md` section 13.
