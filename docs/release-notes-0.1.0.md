# surfbench 0.1.0

First release: a benchmark of implied volatility surface fits (raw SVI, SSVI, eSSVI, SABR
and a quadratic baseline) under one objective and one pre-registered quote filter, scored on
cross-validated accuracy, next-snapshot error and static-arbitrage violations, with
bootstrap intervals and Holm-adjusted pairwise tests.

**All published results are synthetic.** The market-data quick study on Deribit data has not
been run; nothing in this release is a statement about real option markets. REPORT.md
section 3 lists what that study needs (free Tardis.dev downloads, possibly over several daily
quota windows, and about 9 minutes of compute).

## What is included

- Models, arbitrage diagnostics, quote filter (`configs/filter_v1.toml`, frozen
  2026-10-24), Tardis.dev streaming extractor and Deribit live adapter, study runner and
  report, and the `surfbench` CLI (`data`, `fit`, `study`).
- Synthetic validation study (`results/synthetic-v1/`): 24 dates x 2 underlyings from
  arbitrage-free eSSVI, SVI and SABR truths with 0.5 vol point noise. Each model recovers
  its own family at the noise floor; SSVI and eSSVI show zero diagnosed arbitrage; raw SVI
  has the lowest CV RMSE (0.586 BTC, 0.578 ETH vol points), which reflects its flexibility
  on smooth synthetic smiles, not real markets.
- Calibrator check (`results/calibrator-check/`): on 60 noisy synthetic slices the SVI and
  SABR calibrators are never worse than scipy differential evolution (largest relative gap
  1.8e-9).

## Fixed before release

Review of the code first committed on 2026-10-24 found and fixed: the `surfbench.data`
package missing from git and wheels (unanchored ignore rule); eSSVI calendar conditions that
were necessary but not sufficient (now Mingone's sufficient conditions; synthetic results
unchanged to about 1e-8 vol point); several run-provenance and resume issues; Tardis retries
on dropped connections. Details in CHANGELOG.md.

## Changed after the 2026-10-30 changelog entry

- The Tardis extractor ran on the real files of 2019-04, 2020-01 and 2026-10
  (`scripts/extractor_check.py`, `results/extractor-check/`); its layout assumptions and the
  quote conventions hold on them.
- The summary and leaderboard report the data-quality diagnostics and per-step filter counts,
  which were only in the uncommitted unit files. `results/synthetic-v1/` was regenerated from
  the same units; every other number is unchanged.
- `study run` says how many dates are missing and why, exits with 1 while any unit has failed
  (also after its attempts are used up) and with 2 when no snapshot exists; `study report`
  refuses a run without finished units instead of writing an empty leaderboard;
  `summary.json` is strict JSON (`null`, not `NaN`).
- The full study keeps its snapshots in `cache/tardis-60m/`, and the runner refuses Tardis
  snapshots whose times differ from the config. Before, a month cached by the quick study
  would have entered the full study labelled 60-minute although its `t1` was 15 minutes on.
- Removed the unused `plot` extra and leftover CLI scaffolding. The source distribution now
  includes the documents the README links to and the committed results. The build requires
  hatchling 1.26, the first version that reads `license-files` as an array.
- CI: actions pinned to commit SHAs, checkout credentials not persisted.

## Requirements

Python 3.11+, numpy and scipy. Install from GitHub:
`uv tool install git+https://github.com/h-a-forster/surfbench`.

Tardis.dev and Deribit data are not redistributed; download them yourself under the
providers' terms.
