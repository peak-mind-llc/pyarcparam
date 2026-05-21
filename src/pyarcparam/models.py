"""Parametric impulse-response models and goodness-of-fit math.

These are the building blocks shared by both arc-fitting paths:

* **Path A** (``fitting.py``) fits all three candidate models
  (alpha / gamma / log-normal) to the smoothed GFP and selects the best by
  AIC. This yields ``best_model`` and the canonical TCI.
* **Path B** (``decompose.py``) fits *only* a gamma + constant floor,
  iteratively, to provide the arc that per-peak descriptors ride on.

The math is a faithful port of Coherence Workstation's ``self_coherence`` —
no behavioural changes.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as gamma_fn

__all__ = [
    "alpha_func",
    "gamma_func",
    "lognormal_func",
    "gamma_with_floor",
    "gaussian_peak",
    "compute_r_squared",
    "compute_aic",
]


def alpha_func(t, A, tau):
    """Alpha function: single-exponential rise-decay."""
    t_norm = t / tau
    return A * t_norm * np.exp(1.0 - t_norm)


def gamma_func(t, A, n, tau):
    """Gamma function impulse response."""
    return A * (t ** (n - 1) * np.exp(-t / tau)) / (tau**n * gamma_fn(n))


def lognormal_func(t, A, mu, sigma):
    """Log-normal impulse response."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = A * (1.0 / t) * np.exp(-((np.log(t) - mu) ** 2) / (2.0 * sigma**2))
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def gamma_with_floor(t, A, n, tau, C):
    """Gamma impulse response + constant floor.

    ``GFP(t) = A * t^(n-1) * exp(-t/tau) / (tau^n * Gamma(n)) + C``
    """
    return A * (t ** (n - 1) * np.exp(-t / tau)) / (tau**n * gamma_fn(n)) + C


def gaussian_peak(t, amp, center, width):
    """Single Gaussian component peak."""
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)


def compute_r_squared(y_true, y_pred):
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def compute_aic(n, ss_res, k):
    """Akaike Information Criterion.

    Parameters
    ----------
    n : int
        Number of data points.
    ss_res : float
        Sum of squared residuals.
    k : int
        Number of estimated parameters.
    """
    if ss_res <= 0 or n <= 0:
        return float("inf")
    return n * np.log(ss_res / n) + 2 * k
