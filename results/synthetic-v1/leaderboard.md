# Leaderboard: synthetic-v1

Source: `synthetic`. Filter v1 (sha256 `184be3fca2c4`). surfbench 0.1.0, Python 3.13.1, numpy 2.5.4, scipy 1.18.1. Seed 20261023; 10000 bootstrap resamples (dates resampled), 100000 sign flips; next-snapshot horizon 15 min.

Means over dates with 95% percentile bootstrap intervals. Errors in vol points; shares in percent. Arbitrage shares are pooled over slices (pairs) with a date-cluster bootstrap interval.

## BTC

24 dates, 216 slices, 5023 quotes (216 slices / 5023 quotes where every model fit every fold); next snapshot: 5011 quotes, 4927 instruments filtered at both times.

| model | CV RMSE | inside spread % | wing RMSE | next RMSE (common) | butterfly % (wide) | butterfly % (in range) | price check % | calendar % |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.480 [1.164, 1.857] | 77.7 [69.2, 85.4] | 3.552 [2.644, 4.609] | 1.545 [1.236, 1.899] | 3.2 [0.0, 7.4] | 0.0 [0.0, 0.0] | 53.7 [38.4, 68.5] | 41.1 [27.6, 55.2] |
| svi | 0.586 [0.571, 0.600] | 98.4 [98.2, 98.7] | 0.726 [0.658, 0.793] | 0.888 [0.771, 1.015] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.4] | 3.1 [1.0, 5.2] |
| ssvi | 1.420 [1.058, 1.881] | 82.2 [75.0, 88.4] | 2.408 [1.700, 3.309] | 1.624 [1.262, 2.075] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| essvi | 0.701 [0.630, 0.780] | 96.5 [94.7, 97.9] | 1.071 [0.847, 1.320] | 0.977 [0.846, 1.114] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sabr | 0.776 [0.630, 0.977] | 94.5 [89.7, 97.9] | 1.201 [0.847, 1.650] | 1.025 [0.835, 1.250] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Quote persistence (next snapshot, common instruments): 1.028 [0.913, 1.154] vol points.

P(lowest mean CV RMSE): baseline 0.00, svi 1.00, ssvi 0.00, essvi 0.00, sabr 0.00

CV RMSE by true generating family (synthetic data):

| model | essvi | sabr | svi |
|---|---|---|---|
| baseline | 1.015 [0.802, 1.220] | 1.359 [0.841, 1.963] | 2.068 [1.501, 2.784] |
| svi | 0.599 [0.580, 0.618] | 0.585 [0.565, 0.607] | 0.575 [0.539, 0.603] |
| ssvi | 1.064 [0.825, 1.296] | 2.047 [1.064, 3.206] | 1.149 [0.983, 1.371] |
| essvi | 0.555 [0.543, 0.566] | 0.765 [0.624, 0.936] | 0.782 [0.679, 0.904] |
| sabr | 0.595 [0.569, 0.621] | 0.552 [0.533, 0.575] | 1.182 [0.847, 1.628] |

Data quality (medians over dates): our bid IV against the provider's, 0.0000 vol points; put-call-parity forward against the forward used, 0.00e+00 relative gap; filtered `t0` quotes still filtered at `t1`, 98.3%. Quotes left after each filter step at `t0`, summed over dates: input 11830, symbol 11830, age 11830, two_sided 11712, forward 11712, expiry_window 10710, otm 5324, iv 5023, slice 5023.

Exploratory regimes (stress = BTC 30-day ATM IV in the top quartile): 18 calm, 6 stress dates. CV RMSE by regime:

| model | calm | stress |
|---|---|---|
| baseline | 1.389 [1.075, 1.734] | 1.755 [1.033, 2.795] |
| svi | 0.587 [0.568, 0.604] | 0.583 [0.561, 0.606] |
| ssvi | 1.500 [1.039, 2.107] | 1.179 [0.828, 1.527] |
| essvi | 0.726 [0.639, 0.827] | 0.624 [0.555, 0.729] |
| sabr | 0.709 [0.600, 0.845] | 0.977 [0.585, 1.630] |

## ETH

24 dates, 216 slices, 4980 quotes (216 slices / 4980 quotes where every model fit every fold); next snapshot: 4972 quotes, 4891 instruments filtered at both times.

| model | CV RMSE | inside spread % | wing RMSE | next RMSE (common) | butterfly % (wide) | butterfly % (in range) | price check % | calendar % |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.385 [1.123, 1.656] | 78.0 [70.3, 85.3] | 3.308 [2.553, 4.086] | 1.540 [1.294, 1.796] | 2.8 [0.5, 5.6] | 0.0 [0.0, 0.0] | 54.6 [38.9, 70.4] | 46.4 [29.7, 63.5] |
| svi | 0.578 [0.564, 0.594] | 98.8 [98.6, 99.1] | 0.712 [0.656, 0.770] | 0.910 [0.762, 1.084] | 0.5 [0.0, 1.4] | 0.0 [0.0, 0.0] | 1.9 [0.0, 4.2] | 4.7 [1.6, 8.9] |
| ssvi | 1.310 [0.994, 1.680] | 82.7 [74.6, 89.8] | 2.157 [1.563, 2.829] | 1.560 [1.255, 1.901] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| essvi | 0.644 [0.595, 0.700] | 97.5 [96.4, 98.5] | 0.970 [0.782, 1.182] | 0.958 [0.814, 1.129] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sabr | 0.699 [0.619, 0.793] | 96.1 [93.8, 98.1] | 1.141 [0.892, 1.419] | 0.999 [0.837, 1.186] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.6] |

Quote persistence (next snapshot, common instruments): 1.054 [0.918, 1.220] vol points.

P(lowest mean CV RMSE): baseline 0.00, svi 0.99, ssvi 0.00, essvi 0.01, sabr 0.00

CV RMSE by true generating family (synthetic data):

| model | essvi | sabr | svi |
|---|---|---|---|
| baseline | 1.258 [0.807, 1.822] | 1.133 [0.757, 1.510] | 1.763 [1.394, 2.124] |
| svi | 0.580 [0.561, 0.596] | 0.598 [0.570, 0.635] | 0.556 [0.539, 0.572] |
| ssvi | 1.513 [0.765, 2.399] | 1.488 [0.977, 2.019] | 0.929 [0.820, 1.036] |
| essvi | 0.558 [0.545, 0.570] | 0.715 [0.605, 0.832] | 0.660 [0.590, 0.747] |
| sabr | 0.653 [0.575, 0.758] | 0.540 [0.528, 0.552] | 0.903 [0.747, 1.073] |

Data quality (medians over dates): our bid IV against the provider's, 0.0000 vol points; put-call-parity forward against the forward used, 0.00e+00 relative gap; filtered `t0` quotes still filtered at `t1`, 98.4%. Quotes left after each filter step at `t0`, summed over dates: input 11926, symbol 11926, age 11926, two_sided 11831, forward 11831, expiry_window 10737, otm 5345, iv 4980, slice 4980.

Exploratory regimes (stress = BTC 30-day ATM IV in the top quartile): 18 calm, 6 stress dates. CV RMSE by regime:

| model | calm | stress |
|---|---|---|
| baseline | 1.319 [1.024, 1.632] | 1.580 [1.060, 2.096] |
| svi | 0.577 [0.561, 0.598] | 0.580 [0.555, 0.601] |
| ssvi | 1.263 [0.912, 1.702] | 1.453 [0.837, 2.179] |
| essvi | 0.652 [0.592, 0.724] | 0.620 [0.557, 0.700] |
| sabr | 0.674 [0.596, 0.766] | 0.771 [0.571, 1.013] |

## Pairwise comparisons on CV RMSE (primary family, Holm-adjusted)

Mean of date-level differences `a - b` (negative: `a` is more accurate), bootstrap 95% interval, two-sided sign-flip p-value and Holm-adjusted p over all 20 tests.

| underlying | a | b | mean diff | 95% CI | n | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| BTC | baseline | svi | +0.894 | [+0.577, +1.273] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | baseline | ssvi | +0.060 | [-0.299, +0.424] | 24 | 0.7569 | 1.0000 |
| BTC | baseline | essvi | +0.780 | [+0.507, +1.109] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | baseline | sabr | +0.704 | [+0.483, +0.959] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | svi | ssvi | -0.834 | [-1.297, -0.470] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | svi | essvi | -0.115 | [-0.198, -0.040] | 24 | 0.0069 | 0.0552 |
| BTC | svi | sabr | -0.190 | [-0.392, -0.042] | 24 | 0.0182 | 0.1090 |
| BTC | ssvi | essvi | +0.719 | [+0.405, +1.117] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | ssvi | sabr | +0.644 | [+0.239, +1.143] | 24 | 0.0016 | 0.0140 |
| BTC | essvi | sabr | -0.076 | [-0.260, +0.073] | 24 | 0.4595 | 1.0000 |
| ETH | baseline | svi | +0.807 | [+0.541, +1.081] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | baseline | ssvi | +0.074 | [-0.230, +0.382] | 24 | 0.6438 | 1.0000 |
| ETH | baseline | essvi | +0.740 | [+0.501, +0.996] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | baseline | sabr | +0.686 | [+0.481, +0.900] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | svi | ssvi | -0.732 | [-1.099, -0.418] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | svi | essvi | -0.066 | [-0.126, -0.013] | 24 | 0.0303 | 0.1513 |
| ETH | svi | sabr | -0.121 | [-0.219, -0.035] | 24 | 0.0154 | 0.1078 |
| ETH | ssvi | essvi | +0.666 | [+0.369, +1.024] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | ssvi | sabr | +0.612 | [+0.286, +0.982] | 24 | 8.5e-04 | 0.0085 |
| ETH | essvi | sabr | -0.054 | [-0.149, +0.039] | 24 | 0.2844 | 1.0000 |

Dates missing: 0. Units with errors: none.
