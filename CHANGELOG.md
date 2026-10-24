# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Fixed

- Packaging: the `surfbench.data` package (Tardis extractor, Deribit adapter) was excluded
  from git and from built distributions by an unanchored `data/` ignore rule; the rules are
  now anchored to the repository root and a test checks that no package module is ignored.
- eSSVI: the calendar conditions enforced so far were necessary but not sufficient, so the
  fitter could accept crossing slices. The calibration and `check_essvi_pair` now use
  Mingone's (2022) sufficient conditions (both branches). Synthetic results unchanged.
- Study runner: `--force-new` now always restarts; snapshot changes raise a clean mismatch
  error; synthetic snapshots are regenerated when generator inputs change (recorded in
  `meta.json` and the run manifest); `--limit` counts only unfinished units.
- Report: refuses a run directory produced with a different configuration and leaves out
  dates no longer in the configuration.
- Tardis: dropped connections (truncated gzip, `IncompleteRead`) are retried instead of
  marking the month invalid; a full cache stops the session cleanly.

### Added

- Boundary tests for the SVI, SSVI and eSSVI constraint checks, the calendar and
  price-monotonicity diagnostics and the mid-IV computation.

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
  calibration under the Hendriks-Martini calendar conditions, necessary only; see Unreleased), SABR (Hagan lognormal,
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
