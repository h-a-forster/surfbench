# surfbench

An open benchmark of implied volatility surface fits on Deribit BTC and ETH options.

surfbench fits raw SVI, SSVI, eSSVI, SABR and a quadratic baseline to the same filtered quote
snapshots and scores them on three axes:

- **held-out accuracy**: 5-fold cross-validated implied-volatility error on quotes the fit did
  not see, and the share of held-out quotes the fit places inside the bid-ask spread;
- **next-snapshot error**: the fit at `t0` used to price the quotes at `t1 = t0 + h`;
- **static arbitrage**: butterfly violations (Durrleman's `g(k) < 0`, call-price convexity)
  and calendar violations (total variance decreasing in maturity), in and beyond the quoted
  range.

Results are month-level statistics with bootstrap confidence intervals and Holm-adjusted
pairwise comparisons, split into calm and stress months.

> **Status: pre-alpha scaffold.** The design is in [docs/DESIGN.md](docs/DESIGN.md); no
> results exist yet. See [AGENTS.md](AGENTS.md) for the milestone status.

## Data

Snapshots come from the free first-of-month Deribit `options_chain` files published by
[Tardis.dev](https://docs.tardis.dev/historical-data-details/deribit). A single day file is
large (about 18 GB compressed in late 2026), so surfbench streams the start of each file,
keeps only the two snapshot states it needs (a few hundred kilobytes), and stops reading.
Market data is never committed: the repository holds code, configuration, download
metadata (sizes, server checksums, hashes of the extracted snapshots) and aggregate error
statistics only. To reproduce a study you download the data yourself under the provider's
terms.

## Install

```sh
uv sync --group dev
uv run surfbench --version
```

## Planned usage (milestones M4-M6)

```sh
uv run surfbench data tardis --config configs/study_quick.toml   # resumable, quota-aware
uv run surfbench study run    --config configs/study_quick.toml --workers 3
uv run surfbench study report --config configs/study_quick.toml
```

The full study uses `configs/study_full.toml` (every month from 2019-04, 60-minute horizon).

## Licence

MIT. See [LICENSE](LICENSE). Data obtained from Tardis.dev or Deribit is subject to their
own terms and is not covered by this licence.
