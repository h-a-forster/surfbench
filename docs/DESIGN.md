# surfbench design

Status: scaffold (2026-10-24). This document is the specification for v0.1. Formulas,
constraints, reference values and done-checks here are binding for the implementation;
deviations go in section 12 with a reason.

## 1. Goal, headline and claims

**Question.** On liquid crypto options, how do the standard parametric smile and surface
models trade off fit accuracy against static arbitrage, and does the trade-off change in
stress months?

**Headline deliverable (v0.1).** An accuracy-versus-arbitrage leaderboard for raw SVI, SSVI,
eSSVI, SABR and a quadratic baseline on Deribit BTC and ETH first-of-month snapshots:

| axis | metric (per month and underlying) |
|---|---|
| held-out accuracy | 5-fold cross-validated IV RMSE (vol points), inside-spread rate |
| extrapolation | outermost-quote ("wing") IV RMSE |
| next-snapshot | IV RMSE of the `t0` fit on `t1 = t0 + h` quotes, against quote persistence |
| butterfly arbitrage | share of slices with `g(k) < 0` on the quoted range and on a wide grid |
| calendar arbitrage | share of adjacent expiry pairs with `w(k, T2) < w(k, T1)` somewhere |

with month-bootstrap 95% intervals, Holm-adjusted pairwise tests on the primary metric, and
the same table split into calm and stress months.

**What v0.1 will claim, at most.** "On N months of Deribit first-of-month snapshots, model A
has lower cross-validated IV error than model B by X vol points (95% CI), Holm-adjusted p";
"model C produces butterfly violations in Y% of slices (CI)". Arbitrage-free-by-construction
models (SSVI, eSSVI) serve as a check on the diagnostics: any violation they show is a bug
until proven otherwise.

**Non-claims.** No trading or hedging value, no claim about other markets, no claim about
intraday behaviour (one snapshot pair per month at 00:01 UTC), no claim that a model is
"best" beyond the measured metrics and the pre-registered filter.

## 2. Scope review (decisions taken at scaffold time)

The idea as first written (SVI, SSVI, eSSVI, SABR, GP and arbitrage-penalised nets, stress
months, next-snapshot error) does not fit about 4-5 hours of implementation. Cuts and
choices:

1. **Models: SVI, SSVI, eSSVI, SABR (beta = 1) and a quadratic baseline.** GP and neural
   nets are out (training cost, tuning freedom that makes a benchmark unfair, and an
   order-of-magnitude larger validation burden). They are the obvious v0.2 additions
   through the same model protocol.
2. **No maturity interpolation.** All models are evaluated on listed expiries only. The
   calendar check compares adjacent listed expiries. This removes an entire design axis
   (and the Corbetta et al. interpolation scheme becomes unnecessary in v0.1).
3. **One common objective for all models** (section 4.2): weighted least squares on implied
   volatility with half-spread scaling. Papers use different objectives (L1 on prices for
   Corbetta et al.); one objective makes the comparison about the parametrisation, not the
   loss.
4. **Snapshots from the start of each day file only.** Measured on 2026-10-01: the file
   is 17.9 GB compressed, sorted by arrival time, and starts with a full state dump of every
   instrument within one second of midnight. Taking `t0 = 00:01 UTC` and `t1 = t0 + h`
   means streaming only the first `h + 1` minutes (about 10 MB per minute in late 2026) and
   then closing the connection. Intraday snapshots would need gigabytes per month.
5. **Our own implied volatilities.** Bid, ask and mid IVs are recomputed from prices with
   one inversion (section 4.1) for both data sources, because the Deribit live summary
   endpoint does not return bid/ask IVs. Tardis-provided IVs are kept only as a
   data-quality cross-check.
6. **Next-snapshot horizon.** 15 minutes in the quick study, 60 minutes in the full study,
   with quote persistence (the same instrument's `t0` mid IV) as the naive benchmark. A
   short horizon measures surface stability under small moves, not forecasting skill.
7. **Statistics are month-level.** Snapshots within a month share the same day, so the
   month is the independent unit. Inference is on paired month-level differences.
8. **Dependencies: numpy and scipy only** at runtime. No pandas, no parquet: snapshots are a
   few thousand rows and are stored as canonical gzip CSV with a SHA-256.
9. **Wall-time budget.** Compute is small (projected 2-5 minutes for the quick study with 3
   workers). The binding constraint is the Tardis free-download quota (section 5.1), so the
   data step is resumable month by month across sessions and days.

Explicitly deferred: MCS (via `arch`), plots, svi-py cross-implementation comparison as a
study, SSVI joint theta refinement, Obloj's SABR correction, beta = 0.5, intraday snapshots,
Deribit trade data, other exchanges.

## 3. Data licence and handling

- **Tardis.dev terms** (docs.tardis.dev/legal/terms-of-service, read 2026-10-23): data,
  including free sample data, is licensed for internal, research, educational or personal
  use. Redistribution of data or "Manipulated Data" is prohibited (clause 9.2). "Derived
  Data" that does not expose raw records may appear in charts, statistics, research reports
  and papers, but not as a dataset, feed or API (clause 9.3).
- **Deribit terms** (2021 update, article 4.6) restrict market data and derived data to
  personal use without approval.
- **Consequences for this repository.** Nothing that is or reconstructs market data is
  committed: no quotes, no snapshots, no subsampled quote sets, no fitted parameters, no
  IV levels. Committed: code, configuration, download metadata (URL, server-declared size
  and MD5, bytes streamed, SHA-256 of each extracted snapshot file), and aggregate error and
  arbitrage statistics (per month and pooled) plus calm/stress labels. Tests use synthetic
  data only. Live Deribit API snapshots are used for development and for freezing the
  filter, and no results computed from them are committed.
- Before public release, the owner re-reads both terms; if in doubt, publish only pooled
  statistics.

## 4. Conventions and module specifications

Conventions used everywhere:

- `F` forward of the expiry, `K` strike, `k = ln(K/F)`, `T` years to expiry, ACT/365 on a
  24/7 clock (crypto trades continuously). Deribit expiries are 08:00 UTC; Tardis gives
  `expiration` in microseconds, which is used directly.
- Volatility as a decimal; reports show vol points (x100). Total variance `w = sigma^2 T`.
- Zero rates. Deribit inverse options settle in coin: the coin price of an option equals the
  undiscounted Black-76 price with forward `F` divided by `F`. Checked on a live snapshot of
  930 BTC options (2026-10-23): repricing `mark_price` from `mark_iv` with
  `F = underlying_price`, `r = 0` and `T` to 08:00 UTC gives a median absolute error of
  1.7e-6 coin and a maximum of 4.6e-5 (`mark_iv` is rounded to 0.1 vol point).
- Every stochastic function takes `seed`. Simulation p-values use `(1 + count) / (B + 1)`.
- Invalid input raises `ValueError`; statistically undefined results are `nan` with a reason.

### 4.1 `black` (M1)

Normalised (coin) prices with `s = sigma sqrt(T)`, `d1 = -k/s + s/2`, `d2 = d1 - s`:

```
call(k, s) = N(d1) - e^k N(d2)
put(k, s)  = e^k N(-d2) - N(-d1)
vega_s     = e^k n(d2) = n(d1)            (d price / d s)
delta_call = N(d1),  delta_put = N(d1) - 1  (forward deltas, used by the filter)
```

`implied_vol(price, k, T, is_call) -> sigma`: invert for `s` on the OTM side, then
`sigma = s / sqrt(T)`. Valid prices satisfy `0 < price < 1` (call) and `0 < price < e^k`
(put); otherwise return `nan`. Method: vectorised safeguarded Newton-bisection on `s` in
`[1e-8, 20]` (bracket kept, Newton step accepted only inside the bracket), 100 iterations
maximum, tolerance 1e-14 on `s`; `scipy.optimize.brentq` is the reference in tests.

### 4.2 Model protocol (`models.base`, M2)

```python
class SurfaceModel(Protocol):
    name: str

    def fit(self, snap: FitInput) -> FittedSurface: ...


@dataclass(frozen=True)
class FitInput:  # one underlying, one snapshot, training quotes only
    expiries: tuple[float, ...]  # T_j, increasing
    k: tuple[np.ndarray, ...]  # per slice
    iv_mid: tuple[np.ndarray, ...]
    half_spread: tuple[np.ndarray, ...]
    seed: int


class FittedSurface(Protocol):
    def w(self, j: int, k: np.ndarray) -> np.ndarray: ...  # total variance on slice j
    def dw(self, j: int, k: np.ndarray) -> np.ndarray: ...  # analytic or central FD
    def d2w(self, j: int, k: np.ndarray) -> np.ndarray: ...
    def iv(self, j: int, k: np.ndarray) -> np.ndarray: ...  # sqrt(max(w, 0) / T_j)

    params: dict[str, Any]  # kept in run units, never committed
    status: tuple[str, ...]  # per slice: "ok" | "failed:<reason>" | "anchor_lifted"
```

**Common objective.** For every model, the calibration minimises

```
L = sum_i r_i^2,   r_i = (sigma_model(k_i) - sigma_mid_i) / max(h_i, floor)
```

over the training quotes, where `h_i = (ask_iv_i - bid_iv_i) / 2` and `floor = 0.005`
(`weight_floor_iv`). Global models sum over all slices. Models that calibrate in
total-variance space use the first-order equivalent weights
`omega_i = 1 / (2 sigma_mid_i T max(h_i, floor))`, and the reported objective is always the
exact IV-space `L`.

Multi-starts are deterministic (fixed grids; `seed` only breaks ties). Optimiser failure on
a slice sets `status = "failed:..."`; the model does not silently fall back.

### 4.3 Models (M2)

#### 4.3.1 Raw SVI per slice (`models.svi`)

Gatheral (2004); Gatheral and Jacquier (2014), eq. (3.1):

```
w(k) = a + b ( rho (k - m) + sqrt((k - m)^2 + sigma^2) )
w'   = b ( rho + (k - m) / R ),   w'' = b sigma^2 / R^3,   R = sqrt((k - m)^2 + sigma^2)
```

Constraints imposed: `b >= 0`, `|rho| <= 0.999`, `sigma >= 1e-4`, minimum total variance
`a + b sigma sqrt(1 - rho^2) >= 0`, and Lee's moment bound `b (1 + |rho|) <= 2` (Lee 2004;
necessary for no arbitrage). Butterfly and calendar freedom are **not** imposed; measuring
how often unconstrained SVI violates them is part of the benchmark.

Calibration (quasi-explicit, Zeliade 2009): for fixed `(m, sigma)` put `y = (k - m)/sigma`,
`c = b sigma`, `d = rho b sigma`, so `w = a + d y + c sqrt(y^2 + 1)` is linear in `(a, d, c)`.

1. Inner problem: weighted least squares in `(a, d, c)` (weights `omega_i^2`) subject to
   `0 <= c <= 2 sigma`, `|d| <= c`, `|d| <= 2 sigma - c` (Lee) and
   `a + sqrt(c^2 - d^2) >= 0` (minimum variance; a convex constraint). Solve with SLSQP
   from the clipped unconstrained solution. (Zeliade uses `c <= 4 sigma`, i.e. the
   Rogers-Tehranchi slope bound, and `0 <= a <= max w_i`; we use Lee's tighter bound and the
   exact minimum-variance constraint so that negative `a` is allowed.)
2. Outer problem: Nelder-Mead over `(m, sigma)` with `m in [min k - 1, max k + 1]`,
   `sigma in [1e-3, 2]` (clipped), started from the best point of a 5 x 5 grid.
3. Polish: `scipy.optimize.least_squares` on the exact IV residuals over
   `(a, b, rho, m, sigma)` from the quasi-explicit solution, with box bounds and penalty
   residuals for the two nonlinear constraints. Accept only if all constraints hold and `L`
   decreases.

#### 4.3.2 SSVI, global (`models.ssvi`)

Gatheral and Jacquier (2014), eq. (4.1), with the power-law curvature of their Remark 4.4:

```
w(k, theta) = theta/2 ( 1 + rho phi k + sqrt((phi k + rho)^2 + 1 - rho^2) )
phi(theta)  = eta / ( theta^gamma (1 + theta)^(1 - gamma) )
w' = theta phi/2 ( rho + u/R ),  w'' = theta phi^2 (1 - rho^2) / (2 R^3),
u = phi k + rho,  R = sqrt(u^2 + 1 - rho^2)
```

Parameters: shared `(rho, eta, gamma)`; `theta_j` per expiry.

- Static-arbitrage freedom: `theta_j` non-decreasing in `T` (Theorem 4.1), `gamma in
  (0, 1/2]`, `eta (1 + |rho|) <= 2` (Remark 4.4). The restriction `gamma <= 1/2` is needed
  for `theta phi^2 (1 + |rho|) <= 4` as `theta -> 0`, which matters for short crypto expiries
  (`theta` about 1e-3 for 2 days at 50% vol): `theta phi^2 = eta^2 (theta/(1+theta))^(1-2 gamma)
  / (1+theta)` is bounded by `eta^2` for `gamma <= 1/2` and unbounded as `theta -> 0` for
  `gamma > 1/2`.
- Parametrisation: `rho = 0.999 tanh(x1)`, `eta = 2 u / (1 + |rho|)` with
  `u in [0.01, 1]`, `gamma in [0.01, 0.5]`.
- Stage 1: `theta*_j` = mid total variance at `k = 0` by linear interpolation between the two
  training quotes that bracket `k = 0` (nearest quote if not bracketed), then
  `theta_j = max(theta*_j, theta_{j-1})` in increasing `T`.
- Stage 2: `least_squares` over `(rho, u, gamma)` with `theta` fixed, multi-start over
  `rho in {-0.5, 0, 0.3} x u in {0.3, 0.7} x gamma in {0.25, 0.45}`.
- Joint `theta` refinement is deferred (section 2).

#### 4.3.3 eSSVI, slice by slice (`models.essvi`)

Hendriks and Martini (2019): SSVI with maturity-dependent `rho`. Slice `j` in
`(theta, rho, psi)` form, `psi = theta phi`:

```
w(k) = 1/2 ( theta + rho psi k + sqrt((psi k + theta rho)^2 + theta^2 (1 - rho^2)) )
```

- Butterfly (Gatheral-Jacquier Theorem 4.2 in `psi` form, sufficient):
  `psi (1 + |rho|) < 4` and `psi <= 2 sqrt(theta / (1 + |rho|))`.
- Calendar between slices `j-1 < j` (Hendriks-Martini sufficient conditions as used by
  Corbetta et al. 2019): `theta_j >= theta_{j-1}`, `psi_j >= psi_{j-1}`, and
  `|rho_j psi_j - rho_{j-1} psi_{j-1}| <= psi_j - psi_{j-1}`, which is equivalent to
  `psi_j >= psi_minus(rho_j) = max( (psi_p - (rho psi)_p)/(1 - rho_j), (psi_p + (rho psi)_p)/(1 + rho_j) )`.

Calibration (Corbetta, Cohort, Laachir, Martini 2019), slices in increasing `T`:

1. Anchor `(k*, theta*)`: the training quote with smallest `|k|` and its mid total variance;
   `theta = theta* - rho psi k*`, leaving `(rho, psi)` free.
2. For `rho` on 21 points in `[-0.95, 0.95]`: the feasible `psi` interval is
   `0 < psi < min(psi_plus, 4/(1 + |rho|))` with
   `psi_plus = -2 rho k*/(1+|rho|) + sqrt(4 rho^2 k*^2/(1+|rho|)^2 + 4 theta*/(1+|rho|))`,
   `theta > 0`, and for `j > 1`: `psi >= psi_minus(rho)` and `theta >= theta_{j-1}` (a
   half-line in `psi` through `psi_hat = (theta* - theta_{j-1}) / (rho k*)` whose direction
   depends on the sign of `rho k*`; for `rho k* = 0` it requires `theta* >= theta_{j-1}`).
   Minimise `L` over the interval with bounded Brent (`minimize_scalar(method="bounded")`).
3. Refine with 11 points of `rho` on `[rho_best - 0.1, rho_best + 0.1]`.
4. If no `rho` is feasible (market ATM variance below the previous slice), replace
   `theta*` by `theta_{j-1} (1 + 1e-6)`, retry once and set status `anchor_lifted`; if still
   infeasible, `failed:infeasible`.

Every accepted slice is asserted to satisfy the inequalities above (to 1e-12).

#### 4.3.4 SABR per slice (`models.sabr`)

Hagan, Kumar, Lesniewski and Woodward (2002), eq. (2.17a), lognormal implied volatility
with `f = F`, `K` strike:

```
sigma_B = alpha / ( (fK)^((1-b)/2) [1 + (1-b)^2/24 ln^2(f/K) + (1-b)^4/1920 ln^4(f/K)] )
          * z / x(z)
          * [ 1 + ( (1-b)^2/24 alpha^2/(fK)^(1-b) + rho b nu alpha/(4 (fK)^((1-b)/2))
                    + (2 - 3 rho^2)/24 nu^2 ) T ]
z = nu/alpha (fK)^((1-b)/2) ln(f/K),   x(z) = ln( (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) )
```

with `z/x(z) -> 1 - rho z / 2` for `|z| < 1e-7`. `b = beta` is fixed at 1 (config
`sabr_beta`), the usual choice for lognormal-like crypto smiles; then `sigma_B` depends on
`K/F` only. Parameters `alpha in [1e-4, 10]`, `rho in [-0.999, 0.999]`, `nu in [1e-4, 10]`;
`least_squares` with multi-start `alpha0` = nearest-ATM mid IV,
`rho in {-0.5, 0, 0.5} x nu in {0.5, 2}`. Derivatives of `w` by central differences with
step `1e-4`.

#### 4.3.5 Baseline (`models.baseline`)

Per slice `sigma(k) = c0 + c1 k + c2 k^2` by weighted linear least squares (weights
`1/max(h_i, floor)^2`, closed form). Predictions below `1e-4` are clipped and counted.
Analytic derivatives of `w = sigma^2 T`.

### 4.4 Arbitrage diagnostics (`arbitrage`, M3)

Computed on the fit to all `t0` quotes (not the CV folds), per slice `j` with fitted ATM
total variance `theta_j = w_j(0)`:

- **Grids.** Wide: 401 points `k in [-4 sqrt(theta_j), 4 sqrt(theta_j)]`. In-range: the
  points of the wide grid inside `[min k, max k]` of the slice's quotes, plus the quote
  `k` values themselves.
- **Butterfly.** Durrleman's function (Gatheral-Jacquier 2014, Lemma 2.2):

  ```
  g(k) = (1 - k w'/(2w))^2 - (w'^2/4) (1/w + 1/4) + w''/2
  ```

  A slice violates if `w <= 0` or `g < -g_tol` (`g_tol = 1e-6`) at any grid point. Record
  min `g` and the violating `k` interval.
- **Price cross-check.** Normalised call prices `c(k)` on the same grid must be
  non-increasing and convex in `K = F e^k` (second divided differences `>= -1e-10`). Tests
  require agreement with the `g` check on synthetic slices away from the tolerance band.
- **Calendar.** For adjacent listed expiries `T_j < T_{j+1}`, on the wide grid of slice
  `j+1`: violation if `min_k (w_{j+1}(k) - w_j(k)) < -calendar_tol` (`1e-8`); record the
  worst gap. With zero rates and a forward per expiry, `w` non-decreasing in `T` at fixed
  `k` is the no-calendar-arbitrage condition (Gatheral-Jacquier 2014, Lemma 2.1).
- **Slope bounds** (secondary). `|w'(k)| <= 4` on the wide grid (Rogers-Tehranchi);
  Lee's asymptotic `b(1 + |rho|) <= 2` is reported for SVI only.

## 5. Data

### 5.1 Sources (`data.tardis`, `data.deribit`, M4)

**Tardis.dev first-of-month files.** URL
`https://datasets.tardis.dev/v1/deribit/options_chain/{YYYY}/{MM}/01/OPTIONS.csv.gz`, no API
key for the first day of each month. One gzip CSV per day with columns `exchange, symbol,
timestamp, local_timestamp, type, strike_price, expiration, open_interest, last_price,
bid_price, bid_amount, bid_iv, ask_price, ask_amount, ask_iv, mark_price, mark_iv,
underlying_index, underlying_price, delta, gamma, vega, theta, rho` (timestamps in
microseconds; IVs in percent). Inverse BTC/ETH symbols look like `BTC-27NOV26-73000-P`; the
file also holds linear `*_USDC-*` options, which the filter drops.

Measured on 2026-10-23 (file of 2026-10-01): size 17,932,604,919 bytes (`x-dataset-size`
header; an `x-md5` header is also sent); rows strictly ordered by `local_timestamp` (0
inversions in the first 13.4 M rows); all 948 BTC inverse instruments appear within 0.7 s of
midnight (state dump); the first 70 minutes are 687 MB compressed, about 10 MB per minute;
streaming ran at about 7 MB/s.

Observed server behaviour: `HEAD` returns 404 and ranged `GET` returns 403 (Cloudflare), so
the extractor uses plain `GET` only and never `Range`. After about a dozen requests and
about 0.9 GB streamed within roughly ten minutes, every file returned `429` with
`Retry-After` of about 86,100 s (24 hours). The free quota is undocumented.

Extractor algorithm (one month):

1. If `cache/tardis/_quota.json` has `not_before` in the future, stop the session.
2. `GET` with a descriptive `User-Agent`, socket timeout `timeout_s`. Record status, the
   `x-dataset-size` and `x-md5` headers, start time.
3. `404` -> write `meta.json` with `status = "unavailable"`; no retry. `429` -> write
   `_quota.json` with `not_before = now + Retry-After`, stop the session (exit 0 with a
   message). Network errors -> retry the month up to 3 times with 60 s backoff (from the
   start of the file; no ranges).
4. Stream: `zlib.decompressobj(16 + MAX_WBITS)` over 1 MiB chunks, split lines, parse only
   rows whose symbol matches the filter regex, keep `state[symbol] = row`. Validate every
   kept row (24 columns, numeric fields parse, `local_timestamp` non-decreasing; violations
   raise and the month is marked `invalid`).
5. When the first row with `local_timestamp > t0` arrives, freeze a copy of `state` as the
   `t0` snapshot; at the first row `> t1`, freeze `t1` and close the connection.
6. Write `snap_t0.csv.gz`, `snap_t1.csv.gz` (canonical: sorted by symbol, fixed column
   order, gzip `mtime = 0`) and `meta.json` (URL, headers above, bytes streamed, rows read,
   instrument counts at `t0` and `t1`, SHA-256 of each snapshot file, package version)
   atomically (temp file + `os.replace`) under `cache/tardis/YYYY-MM-01/`.
7. Refuse to write if the cache exceeds `cache_cap_gb` (2 GB; snapshots are a few hundred KB,
   so this is a guard, not a budget).

Months are processed in a seeded random order (`fetch_order_seed`), at most
`max_requests_per_session` requests with `request_gap_s` between them, so a quota stop
leaves a random subset rather than a chronological prefix. The full-file MD5 cannot be
verified from a prefix; reproducibility is checked by the snapshot SHA-256 instead.

**Deribit live API** (development and filter freezing only): one call per underlying to
`GET https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency={BTC|ETH}&kind=option`
(about 930 BTC instruments, 2026-10-23). Fields used: `instrument_name`, `bid_price`,
`ask_price`, `mark_price`, `mark_iv`, `underlying_price`, `creation_timestamp` (ms). Expiry
parsed from the name at 08:00 UTC. No bid/ask IVs are returned (we compute them). Optional
second call after `--horizon-min` minutes gives a `t1` snapshot.

### 5.2 Snapshot construction and the pre-registered filter (`quotes`, M4)

`configs/filter_v1.toml` is the filter. It is provisional until M4, when its thresholds are
frozen using live Deribit snapshots only (several hours of the day, never Tardis study
months); the freeze date goes in CHANGELOG. Steps, in order, with counts recorded per step:

1. Symbol regex (inverse BTC/ETH only), quote age `t - timestamp <= 300 s`.
2. Two-sided: `0 < bid < ask`; `bid >= 0.0002` coin.
3. Forward per expiry: median `underlying_price` of the expiry's rows; drop the expiry if
   `(max - min)/median > 0.001`. `k = ln(K/F)`, `T` from `expiration`.
4. Expiry window: `2 <= T*365 <= 550`.
5. OTM side only (calls `K >= F`, puts `K < F`).
6. IVs from bid, ask and mid prices by `black.implied_vol`; drop if any is `nan`, outside
   `[0.05, 5.0]`, or `ask_iv - bid_iv > 0.15`; `|delta| >= 0.02` at the mid IV.
7. Slice kept if `>= 8` quotes and `>= 2` on each side of `k = 0`.

### 5.3 Data-quality diagnostics (reported, not filtered)

Per snapshot: instrument counts per step; median `|our bid IV - Tardis bid_iv|` (expected
well below 0.5 vol point); put-call-parity forward from mid prices against
`underlying_price` (median relative gap); share of `t0` instruments still present at `t1`.

## 6. Study design (M5, M6)

### 6.1 Units and hypotheses

A unit is `(month, underlying)`. The primary family (pre-registered) is all 10 pairwise
differences in `cv_rmse_iv` between the five models, for BTC and ETH separately (20 tests),
Holm at FWER 0.05. Everything else is secondary and reported with unadjusted intervals,
labelled as such.

### 6.2 Procedure per unit (`study.runner`)

1. Load `t0` and `t1` snapshots, apply the filter, record counts.
2. Folds: within each slice sort quotes by `k`; fold of the i-th quote is
   `(i + offset) mod 5`, `offset` drawn from `seed` and the unit id. Every fold spans the
   whole moneyness range, so CV measures interpolation.
3. For each model: one fit per fold on the other four folds (global models fit all slices at
   once); one wing fit without the outermost quote on each side of each slice; one full fit
   on all `t0` quotes. Record time per fit and slice statuses.
4. Arbitrage diagnostics on the full fit (section 4.4).
5. Next snapshot: for expiries present at both times, predict each filtered `t1` quote with
   the full `t0` fit at `k1 = ln(K/F_t1)` on the same expiry (sticky moneyness; implied vol
   as a function of `k` unchanged). Persistence benchmark: the instrument's `t0` mid IV,
   scored on the instruments that pass the filter at both times; models are also scored on
   that common subset.
6. Write the unit JSON through `checkpoint.UnitStore`.

### 6.3 Metrics (`metrics`)

Vol points throughout. Per unit and model: `cv_rmse_iv`, `cv_mae_iv`, `cv_inside_spread`
(share of held-out quotes with `bid_iv <= sigma_hat <= ask_iv`), `cv_rms_spread_units`
(residuals divided by `max(h, floor)`), `wing_rmse_iv`, `fit_rmse_iv`, `next_rmse_iv`,
`next_rmse_iv_common` with `persistence_rmse_iv`, `bfly_viol_wide`, `bfly_viol_inrange`
(shares of slices), `cal_viol` (share of adjacent pairs), worst `g` and worst calendar gap,
`fit_time_s`, `n_failed_slices`.

Failure policy: accuracy metrics are computed on the common set of slices where every model
returned `ok` or `anchor_lifted`; failure counts are reported per model.

### 6.4 Inference and report (`stats`, `study.report`)

- Per model and metric: mean over months with 95% percentile bootstrap intervals
  (B = 10,000, months resampled; moving blocks of 3 months as a robustness check).
- Pairwise: monthly paired differences; two-sided sign-flip randomisation test of the mean
  (100,000 random sign vectors, `p = (1 + count)/(B + 1)`), Holm across the primary family;
  bootstrap interval of the mean difference.
- Rank probabilities: share of bootstrap resamples in which each model has the lowest mean
  `cv_rmse_iv`.
- Arbitrage shares: pooled share of slices (pairs) with a month-cluster bootstrap interval.
- Stress months (exploratory): BTC 30-day ATM IV per month from the stage-1 `theta*` values
  (linear in `T` at 30/365, `sigma = sqrt(theta/T)`); stress = top quartile of the sample
  months (`stress_quantile = 0.75`), used for BTC and ETH alike. Report the leaderboard per
  regime and the regime difference of each pairwise difference with a bootstrap interval;
  no significance claims.
- Report contents: months used and missing (with reason), unit counts, slice and quote counts
  per model, seeds, package and dependency versions, filter version and hash, snapshot
  SHA-256s, wall time. Committed under `results/<name>/` as `summary.json` and
  `leaderboard.md`, with no absolute paths. `REPORT.md` is written by hand from them.

### 6.5 Sizing and wall time

| | quick (`study_quick.toml`) | full (`study_full.toml`) |
|---|---|---|
| months | 28 (Jan/Apr/Jul/Oct 2020-2026) | 91 (2019-04 to 2026-10, minus unavailable) |
| horizon `h` | 15 min | 60 min |
| units | 56 | about 182 |
| fits per unit | 5 models x 7 (5 folds, wing, full) | same |
| stream per month | about 170 MB late 2026, less earlier (projection) | about 640 MB late 2026 |
| download total | projected 2-3 GB | projected 15-30 GB |
| compute | projected 2-5 min with 3 workers | projected 6-15 min with 3 workers |

Projections are replaced by measurements at M5 (timing of 3 real units) and written into
the config headers. Quick study: `uv run surfbench study run --config
configs/study_quick.toml --workers 3`. Full: the same with `configs/study_full.toml`. Both
resume from `runs/<name>/units/`; downloads resume from `cache/tardis/`.

## 7. Validation strategy

### 7.1 Reference values (`scripts/reference_values.py`, independent of the package)

| quantity | value |
|---|---|
| Black-76 call, `F = K = 100`, `T = 1`, `sigma = 0.2` | 7.965567455405804 (QuantLib 1.44 `blackFormula` agrees) |
| call/F, `F = 50000`, `K = 60000`, `T = 0.25`, `sigma = 0.6` | 0.054405634678143 |
| put/F, `F = 50000`, `K = 40000`, `T = 0.25`, `sigma = 0.8` | 0.063911835245144 |
| Vogt raw SVI `(a,b,m,rho,sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)` (G-J Example 3.1) | min `g` = -0.032864 at `k` = 0.8793; `g < 0` on `[0.6425, 1.2569]` (grid step 1e-4) |
| SSVI power law `(eta, rho, gamma)` = (1.2, -0.3, 0.4), (1.5, -0.33, 0.5), (1.0, 0.9, 0.2), `theta` in {0.001 ... 3} | min `g` = 0.2578, 0.2508, 0.2902 (all >= 0) |
| eSSVI slice `(theta, rho, psi) = (0.04, -0.7, 2.6)` | min `g` = -11.930666 (violation) |
| eSSVI `(0.02, -0.4, 0.2) -> (0.05, -0.3, 0.3)` (conditions hold) | min `w2 - w1` on `[-3, 3]` = 2.809898e-2 |
| eSSVI `(0.02, -0.4, 0.2) -> (0.05, 0.6, 0.3)` (ratio 2.6 > 1) | min `w2 - w1` = -2.436786e-1 (calendar violation) |
| SABR Hagan, `F = 1`, `T = 0.25`, `alpha = 0.6`, `rho = -0.2`, `nu = 1.5`, `beta = 1`, `K = 0.7, 1.0, 1.4` | 0.729816776146, 0.619687500000, 0.637525623611 |
| same with `beta = 0.5` | 0.788289252793, 0.623625000000, 0.595862833063 |

The SABR rows agree with QuantLib 1.44 `sabrVolatility` to 12 decimals (checked
2026-10-24). Tests assert these values to 1e-10 (1e-4 for the Vogt interval end points).

### 7.2 Synthetic recovery (`synthetic`, M2 and M5)

`synthetic.make_snapshot(model, params, expiries, strikes, noise, spread, seed)` produces a
`FitInput` plus bid/ask IVs. Checks:

- Own-family, noise-free: each model recovers its surface with IV RMSE below 0.01 vol point.
- Noisy (IV noise 0.5 vol point, spread 2 vol points): all models fit 50 random surfaces
  without exceptions; SSVI and eSSVI fits satisfy their constraints and show zero
  diagnosed violations (an end-to-end check of `arbitrage`).
- A synthetic end-to-end study (6 synthetic months from eSSVI plus noise) runs through
  `study run` and `study report`, resumes after an interrupted run, and the report numbers
  equal a direct recomputation from the unit files.

### 7.3 Checkpointing (`checkpoint`, M1)

`UnitStore(run_dir)`: `units/<unit_id>.json` written atomically, `manifest.json` (config
hash, filter hash, package version, snapshot SHA-256s), `progress.log` (one line per unit
with timestamp, status and elapsed time; no absolute paths). `run_units(units, fn, workers)`
skips finished units, uses at most 3 processes, sets `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to 1 in the parent environment before the pool
starts, records exceptions as `status = "error"` units (message only, paths scrubbed) and
retries them on the next run, up to 3 attempts. A run directory refuses to resume when the
config, filter or snapshot hashes changed (`--force-new` starts over). Long runs are wrapped
in a shell retry loop (AGENTS.md).

### 7.4 Data adapters (offline tests, M4)

A fake opener (or `http.server` on localhost) serves synthetic gzip CSVs in the Tardis
layout: state dump, ordered updates, linear options to be dropped, a malformed row, a
non-monotone timestamp. Tests: snapshot content at `t0` and `t1`, early stop (bytes read
much smaller than the file), 404 -> `unavailable`, 429 -> `_quota.json` and session stop,
timeout -> retry, cache cap, atomic writes, canonical output and stable SHA-256. No test
touches the network.

### 7.5 Third-party cross-checks and bug claims

`scripts/oracles/` (M2, optional, Python 3.13): run with
`uv run --no-project --with QuantLib==1.44 --with svi-py==1.0.0 python scripts/oracles/make_oracles.py`
to write `tests/fixtures/oracles/*.json` (model evaluations on fixed parameters). Tests read
only the JSON. Before calling any third-party result a bug: a minimal repro script, a
check of parameter conventions (total versus implied variance, `psi` versus `phi`, `beta`,
time units), and agreement of our value with the independent reference script.

## 8. Architecture

```
configs/*.toml ──> cli ──> data.tardis / data.deribit ──> cache/<source>/<date>/snap_t{0,1}.csv.gz + meta.json
                     │
                     └──> study.runner ──> quotes (filter, forward, IVs via black)
                                │            └─> FitInput per underlying
                                ├──> models.{baseline, svi, ssvi, essvi, sabr}  (fit, CV, wing, full)
                                ├──> arbitrage (g, price convexity, calendar)
                                ├──> metrics (CV, wing, next-snapshot, persistence)
                                └──> checkpoint.UnitStore ──> runs/<name>/units/*.json
                     study.report ──> stats (bootstrap, sign-flip, Holm) ──> results/<name>/{summary.json, leaderboard.md}
```

| path | role | milestone |
|---|---|---|
| `src/surfbench/black.py` | prices, vega, delta, IV inversion | M1 |
| `src/surfbench/stats.py` | bootstrap, sign-flip test, Holm | M1 |
| `src/surfbench/checkpoint.py` | unit store, progress log, worker pool | M1 |
| `src/surfbench/models/` | protocol and five models | M2 |
| `src/surfbench/synthetic.py` | synthetic snapshots | M2 |
| `src/surfbench/arbitrage.py` | static-arbitrage diagnostics | M3 |
| `src/surfbench/quotes.py` | schema, filter, forwards, slices | M4 |
| `src/surfbench/data/` | Tardis extractor, Deribit live adapter | M4 |
| `src/surfbench/metrics.py` | accuracy metrics, folds | M5 |
| `src/surfbench/study/` | runner and report | M5 |
| `src/surfbench/cli.py` | `data`, `fit`, `study` commands | M4, M5 |
| `scripts/reference_values.py` | independent oracle values (section 7.1) | done |

## 9. Milestones and done-checks

Budget: about 4.5 hours of implementation plus download wall time. Each milestone ends with
`uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` green, an updated
AGENTS.md status section and a commit.

| | scope | done-check | budget |
|---|---|---|---|
| M1 | `black`, `stats`, `checkpoint` | Black rows of 7.1 to 1e-12; IV round trip on `k in [-2, 2]`, `s in [1e-3, 3]` within 1e-10 where `vega_s > 1e-8`, invalid prices give `nan`; brentq agreement; bootstrap/sign-flip/Holm match hand-computed small cases; resume test skips finished units and retries errors | 45 min |
| M2 | `models.*`, `synthetic`, optional oracle fixtures | 7.1 SABR and slice formulas to 1e-10; noise-free own-family recovery < 0.01 vol pt; 50 noisy surfaces fit without exceptions; SSVI/eSSVI constraints asserted on every accepted fit | 75 min |
| M3 | `arbitrage` | Vogt min and interval; admissible SSVI and conditioned eSSVI pairs clean; violating examples flagged; `g` and price checks agree on 200 random SVI slices outside the tolerance band; FD derivatives within 1e-6 of analytic | 35 min |
| M4 | `quotes`, `data.tardis`, `data.deribit`, `data` CLI | 7.4 offline tests; manual live Deribit smoke with step counts; filter frozen and logged; quick-study months fetched as the quota allows (`data status`) | 60 min |
| M5 | `metrics`, `study`, `fit` CLI | 7.2 synthetic end-to-end and resume; report equals recomputation; timing of 3 real units written to config headers | 60 min |
| M6 | quick study, `REPORT.md`, README, CHANGELOG 0.1.0 | `results/quick-v1/` committed; check that no committed file contains quote-level data; REPORT lists sizes, seeds, versions, missing months, intervals and Holm results | 30 min + wall |

## 10. Risks

- **Download quota** (high). The free quota is undocumented and a 24-hour lockout was
  observed after about 0.9 GB. Mitigation: horizon 15 min in the quick study, seeded month
  order, resumable per month, `data status`; the report states exactly which months were
  used. If fewer than 12 quick months are available at M6, report descriptively and say so.
- **Licence** (medium). Only aggregate statistics are committed (section 3); owner review
  before release.
- **Stale wing quotes** (medium). Pre-registered filter (age, two ticks, spread, delta)
  frozen on live data; wing metric reported separately.
- **Forward definition** (medium). `underlying_price` is Deribit's forward per expiry
  (synthetic for expiries without a future); the put-call-parity gap is reported.
- **Thin early years** (medium). 2019-2020 expiries may have few quotes per slice; the
  minimum-quote rule drops them and counts are reported.
- **Implementation bias** (medium). Our SVI/eSSVI calibrators may be weaker than the
  authors'; mitigated by recovery tests, one common objective, and an optional svi-py
  comparison.
- **Short horizon** (low). The next-snapshot metric at 15 minutes mostly measures stability;
  persistence is shown alongside.

## 11. References

- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization
  with application to the valuation of volatility derivatives. Global Derivatives, Madrid.
- Gatheral, J. and Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. Quantitative
  Finance 14(1), 59-71. arXiv:1204.0646.
- Zeliade Systems (2009). Quasi-explicit calibration of Gatheral's SVI model. Zeliade white
  paper.
- Hendriks, S. and Martini, C. (2019). The extended SSVI volatility surface. Journal of
  Computational Finance.
- Corbetta, J., Cohort, P., Laachir, I. and Martini, C. (2019). Robust calibration and
  arbitrage-free interpolation of SSVI slices. Decisions in Economics and Finance 42(2),
  665-677. doi:10.1007/s10203-019-00249-8. arXiv:1804.04924.
- Mingone, A. (2022). No arbitrage global parametrization for the eSSVI volatility surface.
  Quantitative Finance 22(12), 2205-2217. arXiv:2204.00312.
- Pasquazzi, L. (2023). eSSVI surface calibration. arXiv:2304.02106.
- Hagan, P., Kumar, D., Lesniewski, A. and Woodward, D. (2002). Managing smile risk.
  Wilmott Magazine, September, 84-108.
- Lee, R. (2004). The moment formula for implied volatility at extreme strikes.
  Mathematical Finance 14(3), 469-480.
- Rogers, L. C. G. and Tehranchi, M. (2010). Can the implied volatility surface move by
  parallel shifts? Finance and Stochastics 14(2), 235-248.
- Durrleman, V. From implied to spot volatilities. PhD thesis, Princeton University.
- Chataigner, M., Cousin, A., Crepey, S., Dixon, M. and Gueye, D. (2022). Beyond surrogate
  modeling: learning the local volatility via shape constraints. arXiv:2212.09957 (GP and
  NN against SSVI on equity options).
- Matic, J. L., Packham, N. and Hardle, W. K. (2021). Hedging cryptocurrency options.
  arXiv:2112.06807 (SVI surfaces on crypto options as an input to hedging).

## 12. Prior-art software (PyPI, checked 2026-10-23)

| package | version (date) | licence | relevance |
|---|---|---|---|
| svi-py (`pysvi`) | 1.0.0 (2026-10-11) | MIT | SVI, natural SVI, SSVI, eSSVI, JW, SABR calibration library; Python >= 3.13; no benchmark. Optional oracle |
| volsurface | 0.2.0 (2026-03-07) | MIT | raw SVI and SSVI, arbitrage checks, Yahoo data; no benchmark |
| QuantLib | 1.44 (2026-10-13) | BSD-3-Clause | Hagan SABR and Black formula; oracle for 7.1 |
| pysabr | 0.4.1 (2022-04-21) | MIT | Hagan SABR; unmaintained |
| vollib / py_lets_be_rational | 1.0.11 / 1.1.2 (2026) | MIT | Black IV inversion (Jaeckel); not needed for our tolerance |

No open benchmark scoring these models on held-out crypto quotes, arbitrage and
next-snapshot error together was found.

## 13. Implementation notes and deviations

Every deviation from sections 4-7, with its reason.

1. **IV inversion stopping rule (4.1).** When the Newton residual is exactly zero the
   current iterate is kept (the strict bracket test would otherwise force one bisection
   step away from the root). The round-trip check uses out-of-the-money prices, the side
   the inversion works on; in-the-money inputs lose their time value to rounding (for
   example a deep in-the-money call whose time value is 1e-17), so their check is limited
   to time values of at least 1e-6.
2. **SVI start grid (4.3.1).** The 5 x 5 start grid is `m` on 5 equally spaced points over
   `[min k, max k]` and `sigma` on 5 log-spaced points over `[0.01, 1]`; Nelder-Mead then
   searches the full box of section 4.3.1.
3. **eSSVI `rho` polish (4.3.3).** After the 21-point grid and the 11-point refinement, a
   bounded Brent search over `rho` on `[rho_best - 0.02, rho_best + 0.02]` (one refinement
   step), with `psi` re-optimised inside, is accepted when it lowers `L`. Without it the
   `rho` resolution of 0.02 limits noise-free own-family recovery to about 0.2 vol point,
   which fails the 0.01 vol point done-check. The refinement grid is clipped to
   `|rho| <= 0.99`.
4. **SABR numerics (4.3.4).** `x(z)` is evaluated with `log1p` in a cancellation-free form
   on both sides of `z = rho`, and `z/x(z)` uses its third-order series
   `1 - rho z/2 + (1/6 - rho^2/4) z^2 + (5 rho/24 - rho^3/4) z^3` for `|z| < 1e-4`
   instead of the first-order form below 1e-7. The first-order switch made finite-difference
   second derivatives wrong by about 1e-4 near `k = 0`. Where Hagan's expansion is negative
   (large `nu^2 T`), `w = max(sigma_B, 0)^2 T`, and a fit with a non-positive volatility at a
   training quote is reported as `failed:invalid_vol`.
5. **Synthetic API (7.2).** `make_snapshot(surface, grids, noise, spread, seed)` takes any
   fitted surface instead of `(model, params)`; `random_surface(family, rng, expiries)` draws
   admissible parameters per family. Default grids include `k = 0`, so SSVI's stage-1
   `theta*` is exact on noise-free data.
6. **Wide grid covers the quotes (4.4).** The wide grid spans
   `[min(-4 sqrt(theta_j), min k), max(4 sqrt(theta_j), max k)]`, so it always contains the
   in-range grid and `bfly_viol_wide >= bfly_viol_inrange`. For short or low-variance slices
   the quoted range can extend beyond `+-4 sqrt(theta_j)` (Gatheral-Jacquier's Vogt slice
   has `theta = 0.018` and its violation sits at `k` in `[0.64, 1.26]`).
7. **Price cross-check numerics (4.4).** Divided differences use `c = o + (1 - K)^+`, with
   `o` the out-of-the-money price and the intrinsic part differenced exactly. Differencing
   deep in-the-money call prices directly loses about 1e-7 to rounding on the wide grid of
   a long expiry, far above the 1e-10 tolerance.
