# surfbench report

Status on 2026-10-24 (v0.1.0): the full pipeline has been implemented, tested and run end to
end on a **synthetic** validation study. The **market-data quick study has not been run**: the
Tardis.dev free download quota was locked for 24 hours during development (section 3), so no
statement in this report is about Deribit data. Generated tables:
[`results/synthetic-v1/leaderboard.md`](results/synthetic-v1/leaderboard.md) and
[`summary.json`](results/synthetic-v1/summary.json).

## 1. What was run

| item | value |
|---|---|
| study | `configs/study_synthetic.toml` (`synthetic-v1`) |
| dates | 24 synthetic first-of-month dates (2024-01 to 2025-12), two underlyings each: 48 units |
| truth | the generating family rotates eSSVI, raw SVI, SABR (8 dates each); every truth is drawn free of static arbitrage on the diagnostic grids |
| expiries | 10 per unit (1.3 to 363 days; the 1.3-day expiry is removed by the 2-day filter rule) |
| quotes | calls and puts on a fixed 1-2-5 strike grid over +-2.5 ATM standard deviations, coin-priced by Black-76; 0.5 vol point Gaussian noise on mid IVs; full spread 2 vol points at the money, widening by 50% per ATM standard deviation |
| `t1` | 15 minutes later: forwards moved by `N(0, 0.3%)`, all IVs scaled by `1 + N(0, 1%)`, fresh noise, same strikes |
| filter | `configs/filter_v1.toml` (sha256 `098dc3b0f964...`) applied unchanged |
| quotes used | BTC 5,023 and ETH 4,980 `t0` quotes in 216 slices each; every model fit every slice in every fold (0 failed slice fits) |
| inference | 10,000 date-bootstrap resamples, 100,000 sign flips, Holm over 20 tests, seed 20261023 |
| software | surfbench 0.1.0, Python 3.13.1, numpy 2.5.4, scipy 1.18.1 |
| wall time | 492 s with 3 workers (1,370 s of unit compute; mean time per unit and model for its 7 fits: SVI 13.5 s, SABR 6.7 s, SSVI 4.0 s, eSSVI 3.8 s, baseline 0.01 s) |

The synthetic study validates the machinery; it is not evidence about markets. It shows that
(a) every model recovers its own family to the noise floor through the whole quote pipeline,
(b) the arbitrage-free-by-construction models show no diagnosed violations, and (c) the
inference layer produces sensible intervals and multiple-testing control.

## 2. Synthetic results

CV RMSE is the 5-fold cross-validated implied-volatility error in vol points (mean over dates,
95% bootstrap interval). With 0.5 vol point of noise on every mid, the floor for a correctly
specified model is about 0.5 to 0.6 vol point on held-out quotes.

| model | CV RMSE BTC | CV RMSE ETH | wing RMSE BTC | butterfly % BTC (wide) | price check % BTC | calendar % BTC |
|---|---|---|---|---|---|---|
| baseline | 1.48 [1.16, 1.86] | 1.39 [1.12, 1.66] | 3.55 [2.64, 4.61] | 3.2 [0.0, 7.4] | 53.7 [38.4, 68.5] | 41.1 [27.6, 55.2] |
| raw SVI | **0.586 [0.571, 0.600]** | **0.578 [0.564, 0.594]** | 0.73 [0.66, 0.79] | 0.0 | 0.5 [0.0, 1.4] | 3.1 [1.0, 5.2] |
| SSVI | 1.77 [1.26, 2.42] | 1.58 [1.21, 2.00] | 2.87 [1.94, 4.05] | 0.0 | 0.0 | 0.0 |
| eSSVI | 0.780 [0.703, 0.866] | 0.709 [0.649, 0.775] | 1.17 [0.93, 1.44] | 0.0 | 0.0 | 0.0 |
| SABR | 0.776 [0.630, 0.977] | 0.699 [0.619, 0.793] | 1.20 [0.85, 1.65] | 0.0 | 0.0 | 0.0 |

By true family (BTC, CV RMSE): each model is at the noise floor on its own family (eSSVI 0.63
on eSSVI truths, SABR 0.55 on SABR truths, SVI 0.58 on SVI truths), and raw SVI stays at the
floor on all three (0.58 to 0.60). SABR loses on SVI truths (1.18), eSSVI on SVI and SABR
truths (0.88 and 0.84), and SSVI, which shares one `(rho, eta, gamma)` across all expiries
and is not a generating family, is at 1.2 to 2.7 throughout.

Pairwise (primary family, 20 tests, Holm): raw SVI has lower CV RMSE than eSSVI, SSVI and the
baseline on both underlyings (mean differences -0.13 to -1.19 vol point, Holm-adjusted
p <= 0.0006). Against SABR the difference (-0.19 BTC, -0.12 ETH) has a bootstrap interval
that excludes zero but a Holm-adjusted p of 0.092 on both underlyings, so it is not
significant at the family level. eSSVI and SABR are indistinguishable (+0.003 and +0.010
vol point, p = 0.97 and 0.85). SSVI and the baseline are worse than the other three.
P(lowest mean CV RMSE) is 1.00 for raw SVI on both underlyings.

Checks of the machinery:

- **Arbitrage diagnostics.** SSVI and eSSVI show 0 butterfly, 0 price-check and 0 calendar
  violations on both underlyings, as they must. The truths are arbitrage-free, so the
  violations of the other models come from the fits: raw SVI, fit slice by slice, crosses a
  neighbouring slice in 3.1% (BTC) and 4.7% (ETH) of adjacent pairs, and SABR in 0.5% (ETH).
  The baseline fails the price check (call prices rising with strike far in the wings) in
  over half of its slices although its Durrleman `g` is non-negative in about 97% of them.
  This is why the price cross-check has its own column.
- **Next snapshot.** The `t0` fits price `t1` quotes with RMSE 0.89 (SVI) to 1.03 (eSSVI)
  vol points, against 1.03 for quote persistence. With fresh noise at `t1` and a 1% level
  shift, persistence carries the noise of both snapshots, so the models' small edge is the
  expected smoothing effect, not forecasting skill.
- **Intervals.** Moving-block (3-month) intervals on the primary metric are close to the
  i.i.d. ones (SVI BTC 0.573 to 0.598 against 0.571 to 0.600).
- **Calm/stress split.** The synthetic stress label (top quartile of 30-day ATM IV) has no
  built-in effect, and the regime tables show no consistent difference, as expected.

An earlier run of the same study drew SVI and SABR truths slice by slice with independent
parameters. Those truths had calendar arbitrage and Hagan wings beyond Lee's moment bound,
and the arbitrage-free models paid for it (eSSVI CV errors up to 28 vol points on single
dates). That comparison measured the truth's arbitrage rather than the models, so the
generator now draws smooth, arbitrage-free truths (DESIGN section 13, item 5). The run
reported here is the only one with the current generator.

## 3. The market-data study (not run)

Plan: `configs/study_quick.toml`, 28 quarterly first-of-month dates (2020-01 to 2026-10),
BTC and ETH, `t0 = 00:01 UTC`, 15-minute horizon, pre-registered filter v1; commands in
README.md. It was not run because the free Tardis.dev download answered every request with
`429` and a `Retry-After` of about 24 hours after about a dozen research requests on
2026-10-23 around 21:45 UTC; no request may be made before about 2026-10-24 22:00 UTC. The
extractor is tested offline only; its first real run will also be the first check of the
2019-2020 file layout.

Measured compute on real data: four units from two live Deribit snapshot pairs of
2026-10-23 (about 300 filtered quotes in 10 expiries per underlying) took 21 to 43 seconds
each (mean 29 s, one worker on a shared machine), so the quick study should take about 9
minutes and the full study about 30 minutes with 3 workers. Following the data-licence policy (DESIGN section 3), no error or arbitrage
statistic from live snapshots is published.

## 4. The quote filter

`configs/filter_v1.toml` is being reviewed on live Deribit API snapshots (BTC and ETH, a pair
every 30 minutes from 2026-10-23 22:09 UTC) before any study date is examined; no Tardis study
month has been seen. Quote age cannot be checked on live summaries (the endpoint has no
per-instrument quote time), so that rule is applied to Tardis data only.

## 5. Limitations

- **No market-data result yet.** Everything above is synthetic. The synthetic ranking (SVI
  first) mostly reflects flexibility: SVI has five parameters per slice and the truths are
  smooth, so the result says nothing about real smiles, stale quotes or wing behaviour.
- **Synthetic noise** is homoscedastic in the mid and spreads are deterministic; real quotes
  have stale wings, discrete ticks and clustered spreads, which the filter is designed for
  but the synthetic study does not exercise.
- **Calibrator quality.** Our raw SVI (quasi-explicit plus polish) and eSSVI (Corbetta et al.
  with an added `rho` polish, DESIGN section 13) recover their own families to below 1e-7
  vol point without noise. On 60 noisy synthetic slices our SVI and SABR calibrators never
  ended above the objective found by scipy's differential evolution (largest relative gap
  1.8e-9; `results/calibrator-check/`), so they do not get stuck in visibly worse local
  minima there. eSSVI and SSVI were not part of that check, real smiles may be harder, and
  the optional svi-py oracle comparison was not run.
- **Next-snapshot horizon.** 15 minutes measures stability, not forecasting skill.
- **Multiple testing** covers only the 20 primary comparisons; every other number is a
  secondary, unadjusted estimate.
- **Licence.** Only aggregate statistics may be published from the market-data study; the
  owner should re-read the Tardis.dev and Deribit terms before any public release.
