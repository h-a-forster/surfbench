# surfbench

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

An open benchmark of implied volatility surface fits on Deribit BTC and ETH options.

surfbench fits raw SVI, SSVI, eSSVI, SABR and a quadratic baseline to the same filtered quote
snapshots, with one shared objective, and scores them on three axes:

- **held-out accuracy**: 5-fold cross-validated implied-volatility error on quotes the fit did
  not see, the share of held-out quotes placed inside the bid-ask spread, and the error on
  the outermost quote of each smile when it is left out;
- **next-snapshot error**: the fit at `t0` used to price the quotes at `t1 = t0 + h`, next to
  quote persistence as the naive benchmark;
- **static arbitrage**: butterfly violations (Durrleman's `g(k) < 0`, cross-checked on
  call-price monotonicity and convexity) and calendar violations (total variance decreasing
  in maturity), in and beyond the quoted range.

Results are date-level statistics with bootstrap confidence intervals, sign-flip tests with
Holm adjustment across the 20 pairwise model comparisons, and an exploratory calm/stress
split.

Fitting a smile is easy; fitting it without static arbitrage, and so that it generalises to
quotes the fit did not see, is the part that matters for pricing and risk. Papers on SVI,
SSVI and eSSVI each report their own calibrations on their own data with their own loss.
surfbench puts the standard parametrisations side by side on held-out crypto quotes under
one pre-registered quote filter and one objective. To our knowledge there is no open
benchmark that scores these models on held-out quotes, arbitrage and next-snapshot error
together (searched 2026-10-23; see `docs/DESIGN.md` section 12).

> **Status: v0.1.0, pipeline complete, market-data study pending.** Every component is
> implemented and tested, and a 24-month synthetic validation study has run end to end
> ([REPORT.md](REPORT.md)). The quick study on Deribit data waits on the Tardis.dev free
> download quota; no market-data results are published yet.

## Models

| model | scope | parameters | arbitrage-free by construction |
|---|---|---|---|
| quadratic baseline | per expiry | `sigma(k) = c0 + c1 k + c2 k^2` | no |
| raw SVI (Gatheral 2004), quasi-explicit calibration (Zeliade 2009) | per expiry | `a, b, rho, m, sigma` | no (Lee's bound imposed) |
| SSVI, power-law curvature (Gatheral-Jacquier 2014) | global | `rho, eta, gamma` + ATM variance per expiry | yes |
| eSSVI (Hendriks-Martini 2019), robust calibration (Corbetta et al. 2019) | slice by slice | `theta, rho, psi` per expiry | yes (Mingone's sufficient conditions) |
| SABR, Hagan et al. (2002) lognormal, `beta = 1` | per expiry | `alpha, rho, nu` | no |

All models minimise `sum ((sigma_model - sigma_mid) / max(half_spread, 0.5 vol pt))^2` on the
training quotes and are evaluated on listed expiries only. Formulas, constraints and
calibration steps are in [docs/DESIGN.md](docs/DESIGN.md) section 4.

## Install

Not on PyPI. Install from GitHub with uv (or pip):

```sh
uv tool install git+https://github.com/h-a-forster/surfbench   # command-line tool
uv add git+https://github.com/h-a-forster/surfbench            # as a library
```

Python 3.11+. Runtime dependencies: numpy and scipy.

For development:

```sh
git clone https://github.com/h-a-forster/surfbench
cd surfbench
uv sync --group dev
uv run pytest
```

## Quick start

Run the synthetic validation study (no download; 2 to 8 minutes on 3 worker processes):

```sh
uv run surfbench study run    --config configs/study_synthetic.toml --workers 3
uv run surfbench study report --config configs/study_synthetic.toml
cat results/synthetic-v1/leaderboard.md
```

Fit every model to one snapshot (here a synthetic one written by the study above; a live
Deribit snapshot from `surfbench data deribit --out cache/deribit` works the same way):

```sh
uv run surfbench fit cache/synthetic-v1/2024-01-01/snap_t0.csv.gz --underlying BTC
```

```text
BTC: 9 slices, 211 quotes; filter counts {'input': 484, ..., 'slice': 211}
model      fit RMSE   bfly    cal  failed    time
baseline     0.999v   0/9    1/8        0   0.00s
svi          0.431v   0/9    0/8        0   0.90s
ssvi         1.251v   0/9    0/8        0   0.43s
essvi        0.542v   0/9    0/8        0   0.45s
sabr         0.515v   0/9    0/8        0   0.82s
(fit RMSE in vol points on all filtered quotes; bfly = wide-grid butterfly)
```

Use the models as a library:

```python
import numpy as np
from surfbench import arbitrage, synthetic
from surfbench.models import get_model

rng = np.random.default_rng(0)
truth = synthetic.random_surface("essvi", rng)  # five expiries, 1 week to 1 year
snap = synthetic.make_snapshot(truth, noise=0.005, spread=0.02, seed=1)
fit = get_model("svi").fit(snap.data)  # snap.data is a FitInput
fit.iv(0, np.array([-0.1, 0.0, 0.1]))  # implied vols on slice 0
arbitrage.diagnose(fit, snap.data.k)["n_bfly_wide"]  # butterfly-violating slices
```

## The market-data study

Snapshots come from the free first-of-month Deribit `options_chain` files published by
[Tardis.dev](https://docs.tardis.dev/historical-data-details/deribit). A day file is about
18 GB compressed (late 2026), so surfbench streams only its first minutes: the file opens
with a state dump of every instrument at midnight, the extractor keeps the latest row per
option, freezes the state at `t0 = 00:01 UTC` and at `t1 = t0 + h`, and closes the
connection. Each month costs about `h + 1` minutes of stream (about 10 MB per minute in
late 2026).

```sh
uv run surfbench data tardis --config configs/study_quick.toml    # resumable; stops on HTTP 429
uv run surfbench data status --config configs/study_quick.toml
for i in 1 2 3; do uv run surfbench study run --config configs/study_quick.toml --workers 3 && break; done
uv run surfbench study report --config configs/study_quick.toml
```

The quick study covers 28 quarterly dates (January, April, July and October 2020-2026) with
a 15-minute horizon; `configs/study_full.toml` covers every month from 2019-04 with a
60-minute horizon. The free download quota is undocumented: after about a dozen files the
server answered `429` with a 24-hour `Retry-After`. The extractor records the lock in
`cache/tardis/_quota.json`, makes no request until it expires, and fetches months in a seeded
random order so that a partial download is a random subset rather than a chronological
prefix. Compute is about 30 seconds per (date, underlying) unit on one core (measured on 4 live
Deribit units), so the quick study runs in about 9 minutes and the full study in about 30
minutes with 3 workers once the data is cached.

**Data licence.** Tardis.dev and Deribit data are licensed for research and personal use;
redistribution, including manipulated or subsampled data, is not allowed. This repository
holds only code, configuration, download metadata (sizes, server checksums, SHA-256 hashes
of the extracted snapshots) and aggregate error and arbitrage statistics. To reproduce a
study you download the data yourself under the provider's terms.

## Validation

- **Reference values** (`scripts/reference_values.py`, independent of the package): Black-76
  prices (QuantLib 1.44 agrees), Hagan SABR volatilities for `beta = 1` and `0.5` (QuantLib
  1.44 agrees to 12 decimals), the Gatheral-Jacquier Vogt slice (minimum of `g` and the
  violation interval), admissible SSVI parameter sets, and eSSVI butterfly and calendar
  examples. Tests assert the package against these values to 1e-10 where the value is exact.
- **Implied volatility inversion**: round trip within 1e-10 on `k` in `[-2, 2]`,
  `sigma sqrt(T)` in `[1e-3, 3]` wherever vega exceeds 1e-8, and agreement with
  `scipy.optimize.brentq`.
- **Model recovery**: on noise-free synthetic surfaces every model recovers its own family
  with an IV error below 1e-7 vol point (the done-check is 0.01); 50 noisy surfaces fit
  without exceptions; SSVI and eSSVI constraints are asserted on every accepted fit, and
  their fits show zero butterfly, price-check and calendar violations.
- **Calibrator check** (`scripts/calibrator_check.py`, results in
  `results/calibrator-check/`): on one random slice from each of 60 noisy synthetic surfaces,
  our raw SVI and SABR calibrators reach an objective no worse than scipy's differential
  evolution with polishing under the same constraints (largest relative gap 1.8e-9 for SVI,
  1.2e-13 for SABR); SABR is better than differential evolution by more than 1% on 15% of
  slices.
- **Diagnostics**: Durrleman's `g` and the call-price convexity check agree in sign on 200
  random SVI slices; the finite-difference density matches the Breeden-Litzenberger form of
  `g`.
- **Statistics**: sign-flip p-values match exact enumeration; Holm adjustment matches a
  hand-computed case; bootstrap coverage is checked by seeded Monte Carlo.
- **Pipeline**: an end-to-end synthetic study resumes after interruption, and the report's
  numbers equal a direct recomputation from the unit files.
- **Data adapters**: the Tardis extractor is tested offline against synthetic gzip streams
  (state dump, ordered updates, linear options to drop, malformed rows, decreasing
  timestamps, 404, 429, timeouts, cache cap, byte-identical re-extraction).

## Documentation

- [docs/DESIGN.md](docs/DESIGN.md): formulas, constraints, filter, study design, validation
  and every deviation from the original plan (section 13).
- [REPORT.md](REPORT.md): what has been run, with numbers, intervals and limitations.
- [AGENTS.md](AGENTS.md): conventions, architecture map and current status.

## Citation

If you use surfbench, please cite it using [CITATION.cff](CITATION.cff) and the papers of
the models you compare (references in `docs/DESIGN.md` section 11).

## Licence

MIT. See [LICENSE](LICENSE). Data obtained from Tardis.dev or Deribit is subject to their
own terms and is not covered by this licence.
