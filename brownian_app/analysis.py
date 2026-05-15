"""
analysis.py
-----------
Physics layer: MSD fitting, diffusion coefficient, step-size stats.
"""

import numpy as np
from scipy.stats import linregress


def fit_diffusion(t: np.ndarray, msd: np.ndarray) -> dict:
    """
    Fit MSD = 6Dt via linear regression on the middle 20–80% of the curve.

    The trimming avoids two artefacts: early steps have high variance (few
    walkers haven't spread yet), and late steps can show drift from finite
    sample size. The middle segment is where the linear scaling is cleanest.

    Returns dict with keys: D, r_squared, slope, intercept, fit_msd
    """
    n = len(t)
    lo, hi = int(0.20 * n), int(0.80 * n)
    if hi - lo < 5:
        lo, hi = 1, n  # fallback for very short runs

    result    = linregress(t[lo:hi], msd[lo:hi])
    D         = result.slope / 6.0  # 3D: <r²> = 6Dt

    return {
        "D":         D,
        "r_squared": result.rvalue**2,
        "slope":     result.slope,
        "intercept": result.intercept,
        "fit_msd":   result.slope * t + result.intercept,
    }


def final_displacement_stats(displacement: np.ndarray) -> dict:
    """
    Stats on the final displacement of each walker (end-to-end distance).
    theoretical_rms = √N is the expected RMS for an ideal random walk.
    """
    final = displacement[:, -1]
    return {
        "values":          final,
        "mean":            float(final.mean()),
        "std":             float(final.std()),
        "max":             float(final.max()),
        "theoretical_rms": float(np.sqrt(displacement.shape[1] - 1)),
    }
