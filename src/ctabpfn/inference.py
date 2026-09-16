"""
Native TabPFN quantile inference with a content-addressed cache.

`predict_quantiles` fits the pretrained TabPFN regressor and returns its predictive quantiles for each prediction row over the fixed 999-level `GRID`, every tenth of a percent from 0.001 to 0.999, so one inference serves every conformal variant and coverage level. Results are float32 and cached under a sha256 key over the raw arrays, seed, grid, tabpfn version, and device, in a per-model directory `~/.cache/uq-bench/ctabpfn/<MODEL>` by default, since one ctabpfn release pins one tabpfn release and one set of weights; `CTABPFN_CACHE` moves the cache root, setting it empty disables it, and `UQ_BENCH_FAST` disables it too so determinism checks exercise real recomputation. Inference runs on CPU by default for reproducible output, `TABPFN_DEVICE` overrides.
"""

import hashlib
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["GRID", "MODEL", "predict_quantiles"]

GRID: NDArray[np.float64] = np.round(np.linspace(0.001, 0.999, 999), 3)
MODEL = "v3.5"  # the weights this release pins, with tabpfn 9.0.0


def _cache_dir() -> Path | None:
    if os.environ.get("UQ_BENCH_FAST"):
        return None
    override = os.environ.get("CTABPFN_CACHE")
    if override == "":
        return None
    root = (
        Path(override)
        if override is not None
        else Path.home() / ".cache" / "uq-bench" / "ctabpfn"
    )
    return root / MODEL


def _key(arrays: tuple[NDArray, ...], seed: int, device: str) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(f"{array.shape}|{array.dtype}|".encode())
        digest.update(array.tobytes())
    digest.update(f"{seed}|{device}|{version('tabpfn')}|{MODEL}".encode())
    return digest.hexdigest()


def predict_quantiles(
    X_fit: ArrayLike, y_fit: ArrayLike, X_pred: ArrayLike, seed: int
) -> NDArray[np.float32]:
    """Fit TabPFN on the fit rows and return its `GRID` quantiles for each prediction row."""
    arrays = tuple(np.ascontiguousarray(a) for a in (X_fit, y_fit, X_pred))
    device = os.environ.get("TABPFN_DEVICE", "cpu")
    cache = _cache_dir()
    path = cache / f"{_key((*arrays, GRID), seed, device)}.npz" if cache else None
    if path is not None and path.exists():
        with np.load(path) as data:
            return data["quantiles"]
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")  # snapshot at import
    from tabpfn import TabPFNRegressor  # deferred so a cache hit skips the torch import

    model = TabPFNRegressor(device=device, random_state=seed)
    model.fit(arrays[0], arrays[1])
    predicted = model.predict(
        arrays[2], output_type="quantiles", quantiles=[float(q) for q in GRID]
    )
    quantiles = np.stack(predicted, axis=1)
    quantiles = quantiles.astype(np.float32)  # a hit returns a miss's exact bytes
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, quantiles=quantiles)
        os.replace(tmp_path, path)  # atomic publish, no torn parallel reads
    return quantiles
