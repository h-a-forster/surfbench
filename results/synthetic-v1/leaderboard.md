# Leaderboard: synthetic-v1

Source: `synthetic`. Filter v1 (sha256 `098dc3b0f964`). surfbench 0.1.0, Python 3.13.1, numpy 2.5.4, scipy 1.18.1. Seed 20261023; 10000 bootstrap resamples (dates resampled), 100000 sign flips; next-snapshot horizon 15 min.

Means over dates with 95% percentile bootstrap intervals. Errors in vol points; shares in percent. Arbitrage shares are pooled over slices (pairs) with a date-cluster bootstrap interval.

## BTC

24 dates, 216 slices, 5023 quotes (216 slices / 5023 quotes where every model fit every fold); next snapshot: 5011 quotes, 4927 instruments filtered at both times.

| model | CV RMSE | inside spread % | wing RMSE | next RMSE (common) | butterfly % (wide) | butterfly % (in range) | price check % | calendar % |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.480 [1.164, 1.857] | 77.7 [69.2, 85.4] | 3.552 [2.644, 4.609] | 1.545 [1.236, 1.899] | 3.2 [0.0, 7.4] | 0.0 [0.0, 0.0] | 53.7 [38.4, 68.5] | 41.1 [27.6, 55.2] |
| svi | 0.586 [0.571, 0.600] | 98.4 [98.2, 98.7] | 0.726 [0.658, 0.793] | 0.888 [0.771, 1.015] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.4] | 3.1 [1.0, 5.2] |
| ssvi | 1.771 [1.264, 2.418] | 77.8 [70.9, 83.9] | 2.870 [1.939, 4.054] | 1.957 [1.444, 2.599] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| essvi | 0.780 [0.703, 0.866] | 93.6 [91.3, 95.6] | 1.173 [0.933, 1.443] | 1.033 [0.902, 1.172] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sabr | 0.776 [0.630, 0.977] | 94.5 [89.7, 97.9] | 1.201 [0.847, 1.650] | 1.025 [0.835, 1.250] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |

Quote persistence (next snapshot, common instruments): 1.028 [0.913, 1.154] vol points.

P(lowest mean CV RMSE): baseline 0.00, svi 1.00, ssvi 0.00, essvi 0.00, sabr 0.00

CV RMSE by true generating family (synthetic data):

| model | essvi | sabr | svi |
|---|---|---|---|
| baseline | 1.015 [0.802, 1.220] | 1.359 [0.841, 1.963] | 2.068 [1.501, 2.784] |
| svi | 0.599 [0.580, 0.618] | 0.585 [0.565, 0.607] | 0.575 [0.539, 0.603] |
| ssvi | 1.193 [0.949, 1.435] | 2.702 [1.321, 4.323] | 1.418 [1.193, 1.754] |
| essvi | 0.628 [0.617, 0.641] | 0.835 [0.679, 1.026] | 0.876 [0.760, 1.003] |
| sabr | 0.595 [0.569, 0.621] | 0.552 [0.533, 0.575] | 1.182 [0.847, 1.628] |

Exploratory regimes (stress = BTC 30-day ATM IV in the top quartile): 18 calm, 6 stress dates. CV RMSE by regime:

| model | calm | stress |
|---|---|---|
| baseline | 1.389 [1.075, 1.734] | 1.755 [1.033, 2.795] |
| svi | 0.587 [0.568, 0.604] | 0.583 [0.561, 0.606] |
| ssvi | 1.883 [1.224, 2.737] | 1.434 [0.996, 1.929] |
| essvi | 0.807 [0.714, 0.914] | 0.697 [0.610, 0.830] |
| sabr | 0.709 [0.600, 0.845] | 0.977 [0.585, 1.630] |

## ETH

24 dates, 216 slices, 4980 quotes (216 slices / 4980 quotes where every model fit every fold); next snapshot: 4972 quotes, 4891 instruments filtered at both times.

| model | CV RMSE | inside spread % | wing RMSE | next RMSE (common) | butterfly % (wide) | butterfly % (in range) | price check % | calendar % |
|---|---|---|---|---|---|---|---|---|
| baseline | 1.385 [1.123, 1.656] | 78.0 [70.3, 85.3] | 3.308 [2.553, 4.086] | 1.540 [1.294, 1.796] | 2.8 [0.5, 5.6] | 0.0 [0.0, 0.0] | 54.6 [38.9, 70.4] | 46.4 [29.7, 63.5] |
| svi | 0.578 [0.564, 0.594] | 98.8 [98.6, 99.1] | 0.712 [0.656, 0.770] | 0.910 [0.762, 1.084] | 0.5 [0.0, 1.4] | 0.0 [0.0, 0.0] | 1.9 [0.0, 4.2] | 4.7 [1.6, 8.9] |
| ssvi | 1.575 [1.208, 2.001] | 78.2 [70.2, 85.3] | 2.501 [1.819, 3.272] | 1.814 [1.465, 2.206] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| essvi | 0.709 [0.649, 0.775] | 95.7 [94.5, 96.9] | 1.088 [0.882, 1.320] | 1.014 [0.872, 1.178] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| sabr | 0.699 [0.619, 0.793] | 96.1 [93.8, 98.1] | 1.141 [0.892, 1.419] | 0.999 [0.837, 1.186] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | 0.5 [0.0, 1.6] |

Quote persistence (next snapshot, common instruments): 1.054 [0.918, 1.220] vol points.

P(lowest mean CV RMSE): baseline 0.00, svi 1.00, ssvi 0.00, essvi 0.00, sabr 0.00

CV RMSE by true generating family (synthetic data):

| model | essvi | sabr | svi |
|---|---|---|---|
| baseline | 1.258 [0.807, 1.822] | 1.133 [0.757, 1.510] | 1.763 [1.394, 2.124] |
| svi | 0.580 [0.561, 0.596] | 0.598 [0.570, 0.635] | 0.556 [0.539, 0.572] |
| ssvi | 1.614 [0.878, 2.499] | 1.926 [1.155, 2.741] | 1.185 [1.060, 1.312] |
| essvi | 0.598 [0.580, 0.616] | 0.797 [0.675, 0.926] | 0.731 [0.644, 0.833] |
| sabr | 0.653 [0.575, 0.758] | 0.540 [0.528, 0.552] | 0.903 [0.747, 1.073] |

Exploratory regimes (stress = BTC 30-day ATM IV in the top quartile): 18 calm, 6 stress dates. CV RMSE by regime:

| model | calm | stress |
|---|---|---|
| baseline | 1.319 [1.024, 1.632] | 1.580 [1.060, 2.096] |
| svi | 0.577 [0.561, 0.598] | 0.580 [0.555, 0.601] |
| ssvi | 1.539 [1.113, 2.067] | 1.682 [1.031, 2.390] |
| essvi | 0.716 [0.645, 0.799] | 0.687 [0.604, 0.787] |
| sabr | 0.674 [0.596, 0.766] | 0.771 [0.571, 1.013] |

## Pairwise comparisons on CV RMSE (primary family, Holm-adjusted)

Mean of date-level differences `a - b` (negative: `a` is more accurate), bootstrap 95% interval, two-sided sign-flip p-value and Holm-adjusted p over all 20 tests.

| underlying | a | b | mean diff | 95% CI | n | p | p (Holm) |
|---|---|---|---|---|---|---|---|
| BTC | baseline | svi | +0.894 | [+0.577, +1.273] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | baseline | ssvi | -0.291 | [-0.799, +0.150] | 24 | 0.2707 | 1.0000 |
| BTC | baseline | essvi | +0.701 | [+0.434, +1.023] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | baseline | sabr | +0.704 | [+0.483, +0.959] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | svi | ssvi | -1.185 | [-1.832, -0.675] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | svi | essvi | -0.193 | [-0.283, -0.113] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | svi | sabr | -0.190 | [-0.392, -0.042] | 24 | 0.0182 | 0.0924 |
| BTC | ssvi | essvi | +0.991 | [+0.536, +1.573] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | ssvi | sabr | +0.995 | [+0.457, +1.670] | 24 | 1.0e-05 | 2.0e-04 |
| BTC | essvi | sabr | +0.003 | [-0.177, +0.152] | 24 | 0.9743 | 1.0000 |
| ETH | baseline | svi | +0.807 | [+0.541, +1.081] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | baseline | ssvi | -0.190 | [-0.545, +0.144] | 24 | 0.3039 | 1.0000 |
| ETH | baseline | essvi | +0.676 | [+0.438, +0.930] | 24 | 2.0e-05 | 2.0e-04 |
| ETH | baseline | sabr | +0.686 | [+0.481, +0.900] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | svi | ssvi | -0.997 | [-1.420, -0.632] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | svi | essvi | -0.131 | [-0.199, -0.069] | 24 | 8.0e-05 | 5.6e-04 |
| ETH | svi | sabr | -0.121 | [-0.219, -0.035] | 24 | 0.0154 | 0.0924 |
| ETH | ssvi | essvi | +0.866 | [+0.526, +1.269] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | ssvi | sabr | +0.876 | [+0.500, +1.306] | 24 | 1.0e-05 | 2.0e-04 |
| ETH | essvi | sabr | +0.010 | [-0.086, +0.108] | 24 | 0.8477 | 1.0000 |

Dates missing: 0. Units with errors: none.
