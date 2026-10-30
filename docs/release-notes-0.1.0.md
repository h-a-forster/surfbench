# surfbench 0.1.0

First release: a benchmark of implied volatility surface fits (raw SVI, SSVI, eSSVI, SABR
and a quadratic baseline) under one objective and one pre-registered quote filter, scored on
cross-validated accuracy, next-snapshot error and static-arbitrage violations, with
bootstrap intervals and Holm-adjusted pairwise tests.

**All published results are synthetic.** The market-data quick study on Deribit data has not
been run; nothing in this release is a statement about real option markets. REPORT.md
section 3 lists what that study needs (free Tardis.dev downloads over several daily quota
windows, about 9 minutes of compute).

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

## Requirements

Python 3.11+, numpy and scipy. Install from GitHub:
`uv tool install git+https://github.com/h-a-forster/surfbench`.

Tardis.dev and Deribit data are not redistributed; download them yourself under the
providers' terms.
