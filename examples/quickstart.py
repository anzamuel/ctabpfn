"""
Quickstart: conformal TabPFN intervals on California housing.

Calibrates CTabPFN-Q at 90 percent coverage on five random train-test splits of a 3000-row sample and prints per-split and mean test coverage and interval width. The guarantee is marginal, so single splits scatter around the target while the mean over splits sits near it.
"""

import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from ctabpfn import MODEL, quantile_level

X, y = fetch_california_housing(return_X_y=True)
rows = np.random.default_rng(0).permutation(len(y))[:3000]
coverages, widths = [], []
for seed in range(5):
    X_train, X_test, y_train, y_test = train_test_split(
        X[rows], y[rows], test_size=0.2, random_state=seed
    )
    lower, upper, _ = quantile_level(X_train, y_train, X_test, coverage=0.9, seed=seed)
    coverages.append(np.mean((y_test >= lower) & (y_test <= upper)))
    widths.append(np.mean(upper - lower))
    print(f"split {seed}: coverage {coverages[-1]:.3f}, mean width {widths[-1]:.3f}")
print(
    f"TabPFN {MODEL}, mean over splits: coverage {np.mean(coverages):.3f}, width {np.mean(widths):.3f}"
)
