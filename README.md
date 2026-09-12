# ctabpfn
Conformal prediction intervals around the native quantiles of TabPFN, the tabular foundation model. One inference over a dense 999-level quantile grid, every tenth of a percent, serves three split-conformal calibrations, each a single sort with an exact finite-sample coverage guarantee, no grid search and no refitting. The step size bounds the only grid artifact, a one-sided rounding conservatism in the quantile-level variant of at most two steps of coverage, so 0.001 keeps it under 0.2 points, below what benchmark noise can resolve.

## Variants
Every variant splits the incoming train part 75/25 at the given seed, fits TabPFN on the larger part, calibrates on the smaller, and returns lower, upper, and the native median as center.
- `additive`, CTABPFN-A: shifts the native `alpha/2` and `1 - alpha/2` quantiles outward by the order statistic of the CQR score `max(lower - y, y - upper)`; the margin may be negative and tighten the interval.
- `multiplicative`, CTABPFN-M: scales the two half widths around the native median by the order statistic of the ratio score, so the correction adapts to the native local width and may shrink it.
- `quantile_level`, CTABPFN-Q: calibrates the quantile level itself, distributional conformal prediction on the native CDF, choosing the deepest symmetric grid level whose intervals cover enough calibration points. It works at any target coverage, while the other two need `alpha/2` on the grid.

## Usage
```python
from ctabpfn import quantile_level

lower, upper, center = quantile_level(X_train, y_train, X_test, coverage=0.9, seed=0)
```

## Cache
`predict_quantiles` memoizes each TabPFN inference as float32 under `~/.cache/uq-bench/ctabpfn`, keyed by a sha256 over the raw arrays, seed, quantile grid, tabpfn version, and device, so the three variants and every coverage level share one inference per split. `CTABPFN_CACHE` moves the cache, setting it to an empty value disables it, and `UQ_BENCH_FAST` disables it too so determinism checks exercise real recomputation. `TABPFN_DEVICE` selects the inference device, default `cpu` for reproducible output.

## Development
Run `./setup.sh` from the repository root to install dependencies with uv and the pre-commit hooks.
