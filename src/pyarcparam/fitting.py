"""Path A — model selection (3-model, AIC).

Fits alpha, gamma, and log-normal impulse responses to the *smoothed* GFP and
selects the winner by AIC. The R-squared of the winning fit is the canonical
**TCI** (Temporal Coherence Index) and the winner is ``best_model`` — the
"math species" of the cascade (sequential vs. parallel-overlapping).

This is the fit that tells the deck's three-candidate story. It is reported
*separately* from the gamma descriptor arc in :mod:`pyarcparam.decompose`.

Faithful port of Coherence Workstation's ``compute_tci_curvefit`` /
``_fit_tci_single_condition`` (adaptive-smoothing path), reduced to a single
evoked input.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, filtfilt

from .models import (
    alpha_func,
    compute_aic,
    compute_r_squared,
    gamma_func,
    lognormal_func,
)

__all__ = ["fit_all_models", "fit_tci"]

_PARAM_NAMES = {
    "alpha": ["A", "tau"],
    "gamma": ["A", "n", "tau"],
    "log_normal": ["A", "mu", "sigma"],
}

# Adaptive cutoff constants (Hz)
_INITIAL_CUTOFF_HZ = 5.0
_CUTOFF_MULTIPLIER = 7.5
_CUTOFF_MIN_HZ = 1.5
_CUTOFF_MAX_HZ = 5.0
_FALLBACK_CUTOFF_HZ = 3.0

_MODEL_FUNCS = {
    "alpha": (alpha_func, ["A", "tau"]),
    "gamma": (gamma_func, ["A", "n", "tau"]),
    "log_normal": (lognormal_func, ["A", "mu", "sigma"]),
}


def _fit_model(name, func, t, y, p0, bounds, n_params):
    """Fit a single candidate model and return metrics.

    Returns a dict with ``r_squared, aic, params, y_pred``. On failure,
    ``r_squared=0, aic=inf, params={}``.
    """
    fail = {"r_squared": 0.0, "aic": float("inf"), "params": {}, "y_pred": None}
    try:
        popt, _ = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=5000)
        y_pred = func(t, *popt)
        r2 = compute_r_squared(y, y_pred)
        ss_res = float(np.sum((y - y_pred) ** 2))
        aic = compute_aic(len(y), ss_res, n_params)
        params = {k: float(v) for k, v in zip(_PARAM_NAMES[name], popt, strict=True)}
        return {
            "r_squared": float(r2),
            "aic": float(aic),
            "params": params,
            "y_pred": y_pred,
        }
    except Exception:
        return fail


def _lowpass_filter_gfp(gfp, sfreq, cutoff=3.0, order=4):
    """Zero-phase Butterworth low-pass filter for the GFP envelope."""
    nyq = sfreq / 2.0
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, gfp)


def fit_all_models(t_fit, gfp_post_uv, peak_amp, peak_time):
    """Fit all three candidate impulse-response models.

    Returns ``(model_fits dict, best_name, best_fit)`` where the winner is
    chosen by minimum AIC.
    """
    model_fits = {}

    model_fits["alpha"] = _fit_model(
        "alpha",
        alpha_func,
        t_fit,
        gfp_post_uv,
        p0=[peak_amp, peak_time],
        bounds=([0, 0.01], [peak_amp * 10, 0.5]),
        n_params=2,
    )

    model_fits["gamma"] = _fit_model(
        "gamma",
        gamma_func,
        t_fit,
        gfp_post_uv,
        p0=[peak_amp * 5, 3.0, peak_time / 3],
        bounds=([0, 1, 0.01], [peak_amp * 50, 20, 0.5]),
        n_params=3,
    )

    mu_init = np.log(max(peak_time, 0.01))
    model_fits["log_normal"] = _fit_model(
        "log_normal",
        lognormal_func,
        t_fit,
        gfp_post_uv,
        p0=[peak_amp * peak_time, mu_init, 0.5],
        bounds=([0, -3, 0.1], [peak_amp * peak_time * 20, 1, 2.0]),
        n_params=3,
    )

    best_name = None
    best_aic = float("inf")
    for name, fit in model_fits.items():
        if fit["aic"] < best_aic:
            best_aic = fit["aic"]
            best_name = name

    best_fit = model_fits[best_name] if best_name else None
    return model_fits, best_name, best_fit


def _estimate_arc_natural_frequency(model_name, params_dict):
    """Estimate the natural frequency of a fitted arc via autocorrelation.

    Generates the fitted impulse response at 1000 Hz over 0-2 s, computes its
    autocorrelation, and finds the first zero crossing. The natural frequency
    is ``1 / (4 * t_zero_crossing)``. Returns Hz, or ``None`` on failure.
    """
    if model_name not in _MODEL_FUNCS:
        return None

    func, param_order = _MODEL_FUNCS[model_name]
    param_values = [params_dict[k] for k in param_order]

    fs_synth = 1000.0
    t_synth = np.arange(0, 2.0, 1.0 / fs_synth)
    t_synth[0] = 1.0 / fs_synth  # avoid t=0 for log-normal (1/t)

    try:
        y_synth = func(t_synth, *param_values)
    except Exception:
        return None

    if np.all(y_synth == 0) or np.any(np.isnan(y_synth)):
        return None

    y_centered = y_synth - np.mean(y_synth)
    autocorr = np.correlate(y_centered, y_centered, mode="full")
    autocorr = autocorr[len(autocorr) // 2 :]
    if autocorr[0] != 0:
        autocorr = autocorr / autocorr[0]

    for i in range(1, len(autocorr)):
        if autocorr[i] <= 0:
            denom = autocorr[i - 1] - autocorr[i]
            if denom == 0:
                break
            t_zero = (i - 1 + autocorr[i - 1] / denom) / fs_synth
            if t_zero > 0:
                return float(1.0 / (4.0 * t_zero))
            break

    return None


def _serialize_model_fits(model_fits):
    """Strip ``y_pred`` and handle inf AIC for JSON serialization."""
    return {
        name: {
            "r_squared": fit["r_squared"],
            "aic": fit["aic"] if not np.isinf(fit["aic"]) else None,
            "params": fit["params"],
        }
        for name, fit in model_fits.items()
    }


def fit_tci(evoked):
    """Compute the Temporal Coherence Index for a single evoked response.

    Two-pass adaptive-smoothing approach:

    1. Initial fit at 5 Hz cutoff to estimate the arc shape.
    2. Derive a per-condition cutoff from the autocorrelation of the fitted
       curve (``f_natural * 7.5``, clamped to [1.5, 5.0] Hz).
    3. Re-smooth and re-fit at the derived cutoff for the final result.

    Returns a dict with ``tci, best_model, peak_latency_ms, amplitude_uv,
    model_fits, smoothing_cutoff_hz, natural_frequency_hz,
    adaptive_cutoff_method, fit_failed``.
    """
    sfreq = float(evoked.info["sfreq"])
    times = np.asarray(evoked.times)  # seconds

    gfp_raw = np.std(evoked.data, axis=0)  # V

    post_mask = times >= 0
    t_post = times[post_mask]
    if len(t_post) == 0:
        return _empty_tci()

    # t_fit avoids t=0 for log-normal (1/t singularity)
    t_fit = t_post.copy()
    if t_fit[0] <= 0:
        t_fit[0] = 1.0 / sfreq

    # Pass 1: initial fit at permissive cutoff
    smoothed_pass1 = _lowpass_filter_gfp(gfp_raw, sfreq, cutoff=_INITIAL_CUTOFF_HZ)
    gfp_post_uv_1 = smoothed_pass1[post_mask] * 1e6

    peak_idx_1 = int(np.argmax(gfp_post_uv_1))
    peak_time_1 = max(t_fit[peak_idx_1], 0.02)
    peak_amp_1 = gfp_post_uv_1[peak_idx_1]
    if peak_amp_1 <= 0:
        peak_amp_1 = 1.0

    _, best_name_1, best_fit_1 = fit_all_models(t_fit, gfp_post_uv_1, peak_amp_1, peak_time_1)
    pass1_ok = (
        best_fit_1 is not None
        and best_fit_1["y_pred"] is not None
        and best_fit_1["r_squared"] >= 0.30
    )

    # Derive adaptive cutoff
    natural_freq = None
    adaptive_method = "fallback"
    derived_cutoff = _FALLBACK_CUTOFF_HZ
    if pass1_ok:
        natural_freq = _estimate_arc_natural_frequency(best_name_1, best_fit_1["params"])
        if natural_freq is not None:
            derived_cutoff = float(
                np.clip(natural_freq * _CUTOFF_MULTIPLIER, _CUTOFF_MIN_HZ, _CUTOFF_MAX_HZ)
            )
            adaptive_method = "autocorrelation"

    # Pass 2: re-smooth and re-fit at derived cutoff
    smoothed_pass2 = _lowpass_filter_gfp(gfp_raw, sfreq, cutoff=derived_cutoff)
    gfp_post_uv = smoothed_pass2[post_mask] * 1e6

    peak_idx = int(np.argmax(gfp_post_uv))
    peak_time = max(t_fit[peak_idx], 0.02)
    peak_amp = gfp_post_uv[peak_idx]
    if peak_amp <= 0:
        peak_amp = 1.0

    model_fits, best_name, best_fit = fit_all_models(t_fit, gfp_post_uv, peak_amp, peak_time)

    cutoff_meta = {
        "smoothing_cutoff_hz": derived_cutoff,
        "natural_frequency_hz": (float(natural_freq) if natural_freq is not None else None),
        "adaptive_cutoff_method": adaptive_method,
    }

    fit_failed = best_fit is None or best_fit["y_pred"] is None or best_fit["r_squared"] < 0.30
    if fit_failed:
        return {
            "tci": None,
            "best_model": None,
            "peak_latency_ms": None,
            "amplitude_uv": None,
            "model_fits": _serialize_model_fits(model_fits),
            "fit_failed": True,
            **cutoff_meta,
        }

    y_pred = best_fit["y_pred"]
    fitted_peak_idx = int(np.argmax(y_pred))
    return {
        "tci": float(best_fit["r_squared"]),
        "best_model": best_name,
        "peak_latency_ms": float(t_post[fitted_peak_idx] * 1e3),
        "amplitude_uv": float(y_pred[fitted_peak_idx]),
        "model_fits": _serialize_model_fits(model_fits),
        "fit_failed": False,
        **cutoff_meta,
    }


def _empty_tci():
    return {
        "tci": None,
        "best_model": None,
        "peak_latency_ms": None,
        "amplitude_uv": None,
        "model_fits": {},
        "fit_failed": True,
        "smoothing_cutoff_hz": _FALLBACK_CUTOFF_HZ,
        "natural_frequency_hz": None,
        "adaptive_cutoff_method": "fallback",
    }
