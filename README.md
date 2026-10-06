# ctabpfn
Split-conformal prediction intervals for regression, built on the native predictive quantiles of TabPFN. By Samuel Anzalone and Jakob Heiss. Evaluated with [uq-bench](https://github.com/anzamuel/uq-bench).

Paper: [CTabPFN: Conformal Uncertainty Quantification with TabPFN](paper/ctabpfn.pdf), semester project, ETH Zurich, 2026. The main results use TabPFN 3; Appendix C repeats the benchmark on TabPFN-3.5.

## Method
One TabPFN forward pass over the fitting split returns, for every calibration and test point, the quantiles $\hat q_\tau(x)$ on the grid $\tau \in \{0.001, 0.002, \ldots, 0.999\}$, with median $\hat m = \hat q_{0.5}$. Three calibrations share this inference, and none refits TabPFN. With miscoverage $\alpha = 1 - \text{coverage}$ and $n_1$ calibration points, A and M take $\hat t$ as the $\lceil (1-\alpha)(n_1+1) \rceil$-th smallest score.

| function | variant | score $s_i$ | interval |
| --- | --- | --- | --- |
| `additive` | CTabPFN-A | $\max(\hat q_{\alpha/2}(X_i) - Y_i,\ Y_i - \hat q_{1-\alpha/2}(X_i))$ | $[\hat q_{\alpha/2} - \hat t,\ \hat q_{1-\alpha/2} + \hat t]$ |
| `multiplicative` | CTabPFN-M | $\max\left(\frac{\hat m - Y_i}{\hat m - \hat q_{\alpha/2}},\ \frac{Y_i - \hat m}{\hat q_{1-\alpha/2} - \hat m}\right)$ | $[\hat m - \hat t(\hat m - \hat q_{\alpha/2}),\ \hat m + \hat t(\hat q_{1-\alpha/2} - \hat m)]$ |
| `quantile_level` | CTabPFN-Q | largest grid $\tau$ with $Y_i \in [\hat q_\tau, \hat q_{1-\tau}]$ | $[\hat q_{\hat\tau},\ \hat q_{1-\hat\tau}]$ |

For Q, $\hat\tau$ is the $\lfloor \alpha(n_1+1) \rfloor$-th smallest score, so both endpoints are native quantiles. The A margin can be negative and the M factor below one, so both can shrink the native interval. A and M need $\alpha/2$ on the grid, Q works at any coverage. Each function splits its training data at `seed`, fits on a `fit_fraction` share, default 0.9, and calibrates on the rest. It returns `lower`, `upper`, and the native median as `center`. A smaller fraction gives steadier per-split coverage on small datasets at the cost of wider intervals.

## Quickstart with TabPFN-3.5
```bash
git clone -b v3.5 https://github.com/anzamuel/ctabpfn && cd ctabpfn
export TABPFN_TOKEN=<your key>  # free Prior Labs account, accept the TabPFN license at https://ux.priorlabs.ai
uv run examples/quickstart.py   # calibrates 90 percent intervals on five California-housing splits
```
The `v3.5` branch, tagged `v0.2.0`, pins tabpfn 9.0.0 with the TabPFN-3.5 weights, downloaded on first use. `main` and `v0.1.0` pin tabpfn 8.5.0 with the v3 weights. Only [uv](https://docs.astral.sh/uv/) is required.

## Install
```bash
uv add "ctabpfn @ git+https://github.com/anzamuel/ctabpfn@v0.2.0"  # TabPFN-3.5
uv add "ctabpfn @ git+https://github.com/anzamuel/ctabpfn@v0.1.0"  # TabPFN v3
```
```python
from ctabpfn import quantile_level

lower, upper, center = quantile_level(X_train, y_train, X_test, coverage=0.9, seed=0)
```
CPU inference is deterministic and handles a few thousand training rows. Set `TABPFN_DEVICE=cuda` or `mps` for larger data.

## Guarantee
If the training points and the test point are exchangeable, for instance i.i.d., all three variants satisfy

$$\mathbb{P}\big(Y_{n+1} \in [\text{lower}(X_{n+1}), \text{upper}(X_{n+1})]\big) \ge 1 - \alpha$$

for any data distribution and any TabPFN version. The probability is over the training data, the seeded fit-calibration split, and the test point. Coverage is marginal. It holds on average, not conditionally on $x$ and not for a single fixed split. If the scores are almost surely distinct, A and M also cover at most $1 - \alpha + 1/(n_1+1)$. The finite grid adds a one-sided caveat to Q. Rounding scores to the step $\delta = 0.001$ can raise its coverage by at most $2\delta = 0.002$ above a continuous level, and never lowers it.

## Results
From [uq-bench](https://github.com/anzamuel/uq-bench): 12 regression datasets, 10 seeds each, 80/20 train-test splits, target coverage 0.9. PICP is test coverage. NIW is mean interval width divided by the test response range. Both are averaged over all 120 runs.

| method | PICP | NIW |
| --- | --- | --- |
| split conformal | 0.903 | 0.320 |
| UACQR-P | 0.904 | 0.188 |
| PCS-UQ | 0.900 | 0.127 |
| CLEAR | 0.899 | 0.136 |
| CTabPFN-A, v3 / v3.5 | 0.899 / 0.898 | 0.087 / 0.083 |
| CTabPFN-M, v3 / v3.5 | 0.898 / 0.898 | 0.081 / 0.079 |
| CTabPFN-Q, v3 / v3.5 | 0.899 / 0.899 | 0.079 / 0.075 |
| native TabPFN, v3 / v3.5, no guarantee | 0.911 / 0.926 | 0.077 / 0.082 |

Native TabPFN v3 coverage ranges from 0.875 to 1.000 across datasets. Per dataset, the median width of the best CTabPFN v3 variant is 0.75 of the best retrained baseline. In uq-bench this package runs as the methods `ctabpfn-{a,m,q}-v3` and `ctabpfn-{a,m,q}-v3.5`. The datasets are public Parquet tables at https://huggingface.co/datasets/anzamuel/uq-bench. The quickstart fetches California housing through scikit-learn.

## Cache
`predict_quantiles` stores each TabPFN inference as float32 under `~/.cache/uq-bench/ctabpfn/<MODEL>`. The key is a sha256 over the raw arrays, seed, quantile grid, tabpfn version, device, and model. The three variants and all coverage levels therefore share one inference per split, and model generations never collide. `CTABPFN_CACHE` moves the cache root, and an empty value disables it. `UQ_BENCH_FAST` also disables it, so determinism checks recompute. `TABPFN_DEVICE` selects the device, default `cpu` for reproducible output.

## Versioning
Each ctabpfn release pins one tabpfn release and one set of weights, exposed as `ctabpfn.MODEL`. The `v3.5` branch, release `v0.2.0`, ships `v3.5` on tabpfn 9.0.0. `main`, release `v0.1.0`, ships `v3` on tabpfn 8.5.0, the configuration of the paper. A new TabPFN generation gets a new release. Each [uq-bench](https://github.com/anzamuel/uq-bench) method folder pins one ctabpfn commit and names the model, for instance `ctabpfn-q-v3.5`.

## Development
Run `./setup.sh` from the repository root to install dependencies with uv and the pre-commit hooks.

## References
Compared methods: split conformal [1, 2], UACQR-P [5], PCS-UQ [6], CLEAR [7], and native TabPFN [8, 9]. The A score is the CQR score [3], also applied to TabPFN in [10], and Q follows distributional conformal prediction [4]. TabPFN-3.5 has no published reference yet.

1. V. Vovk, A. Gammerman, G. Shafer. Algorithmic Learning in a Random World. Springer, 2005.
2. J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, L. Wasserman. Distribution-Free Predictive Inference for Regression. Journal of the American Statistical Association, 113(523):1094-1111, 2018.
3. Y. Romano, E. Patterson, E. J. Candès. Conformalized Quantile Regression. Advances in Neural Information Processing Systems 32, 2019.
4. V. Chernozhukov, K. Wüthrich, Y. Zhu. Distributional Conformal Prediction. Proceedings of the National Academy of Sciences, 118(48), 2021. [arXiv:1909.07889](https://arxiv.org/abs/1909.07889)
5. R. Rossellini, R. F. Barber, R. Willett. Integrating Uncertainty Awareness into Conformalized Quantile Regression. International Conference on Artificial Intelligence and Statistics, 2024. [arXiv:2306.08693](https://arxiv.org/abs/2306.08693)
6. A. Agarwal, F. Xiao, R. Barter, O. Ronen, B. Fan, B. Yu. PCS-UQ: Uncertainty Quantification via the Predictability-Computability-Stability Framework. arXiv preprint, 2025. [arXiv:2505.08784](https://arxiv.org/abs/2505.08784)
7. I. Azizi, J. Bodik, J. Heiss, B. Yu. CLEAR: Calibrated Learning for Epistemic and Aleatoric Risk. arXiv preprint, 2026. [arXiv:2507.08150](https://arxiv.org/abs/2507.08150)
8. N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, F. Hutter. Accurate Predictions on Small Data with a Tabular Foundation Model. Nature, 637:319-326, 2025. [doi:10.1038/s41586-024-08328-7](https://doi.org/10.1038/s41586-024-08328-7)
9. L. Grinsztajn, K. Flöge, O. Key, F. Birkel, P. Jund, B. Roof, et al. TabPFN-3: Technical Report. arXiv preprint, 2026. [arXiv:2605.13986](https://arxiv.org/abs/2605.13986)
10. F. D. van Leeuwen. Conformal Prediction for Tabular Prior-Data Fitted Networks with Missing Data. OpenReview preprint, 2025. [openreview:TnZcC7GXI5](https://openreview.net/forum?id=TnZcC7GXI5)

## Citation
Samuel Anzalone and Jakob Heiss, [CTabPFN: Conformal Uncertainty Quantification with TabPFN](paper/ctabpfn.pdf), semester project, ETH Zurich, 2026.
```bibtex
@misc{anzalone2026ctabpfn,
  title        = {{CTabPFN}: Conformal Uncertainty Quantification with {TabPFN}},
  author       = {Anzalone, Samuel and Heiss, Jakob},
  year         = {2026},
  howpublished = {Semester project, ETH Zurich},
  url          = {https://github.com/anzamuel/ctabpfn/blob/main/paper/ctabpfn.pdf}
}
```

## License
Apache 2.0, see [LICENSE](LICENSE).
