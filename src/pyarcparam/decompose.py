"""Path B — the descriptor arc (gamma-only, iterative).

This is the arc that *all per-peak descriptors are measured against*. It is a
faithful port of Coherence Workstation's ``iterative_decompose``:

1. Fit gamma + constant floor to the (possibly peak-subtracted) GFP.
2. Find peaks in the residual against the *original* GFP.
3. Subtract those peaks and re-fit.
4. Repeat until the time constant tau stabilizes.

Note that this path fits **gamma only** — never alpha or log-normal. Even when
the 3-model AIC selection in :mod:`pyarcparam.fitting` reports a different
``best_model``, the descriptor arc here is always a gamma. This matches the
shipping product; see the README "Two fits, on purpose" section.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .models import compute_r_squared, gamma_with_floor, gaussian_peak

__all__ = [
    "fit_gamma_arc",
    "find_residual_peaks",
    "build_component_waveform",
    "iterative_decompose",
]


def fit_gamma_arc(t_s, gfp, peak_amp, peak_time, fixed_tau=None):
    """Fit gamma + floor to the GFP arc.

    If *fixed_tau* is provided, tau is locked and only A, n, C are free.

    Returns a dict with ``A, n, tau, C, r2, y_pred, tau_at_ceiling,
    tau_fixed`` or ``None`` on failure.
    """
    epoch_duration = float(t_s[-1] - t_s[0])
    tau_max = epoch_duration / 2.0
    ceiling_threshold = tau_max * 0.95

    # Estimate floor from the last 20% of the epoch
    tail_start = int(len(gfp) * 0.8)
    floor_est = float(np.median(gfp[tail_start:]))
    floor_est = max(floor_est, 0.0)

    if fixed_tau is not None:

        def _model(t, A, n, C):
            return gamma_with_floor(t, A, n, fixed_tau, C)

        try:
            popt, _ = curve_fit(
                _model,
                t_s,
                gfp,
                p0=[peak_amp * 5, 3.0, floor_est],
                bounds=([0, 1, 0], [peak_amp * 50, 20, peak_amp]),
                maxfev=10000,
            )
            y_pred = gamma_with_floor(t_s, popt[0], popt[1], fixed_tau, popt[2])
            r2 = compute_r_squared(gfp, y_pred)
            return {
                "A": float(popt[0]),
                "n": float(popt[1]),
                "tau": float(fixed_tau),
                "C": float(popt[2]),
                "tau_fixed": True,
                "tau_at_ceiling": False,
                "r2": float(r2),
                "y_pred": y_pred,
            }
        except Exception:
            return None
    else:
        try:
            popt, _ = curve_fit(
                gamma_with_floor,
                t_s,
                gfp,
                p0=[peak_amp * 5, 3.0, peak_time / 3, floor_est],
                bounds=(
                    [0, 1, 0.005, 0],
                    [peak_amp * 50, 20, tau_max, peak_amp],
                ),
                maxfev=10000,
            )
            y_pred = gamma_with_floor(t_s, *popt)
            r2 = compute_r_squared(gfp, y_pred)
            at_ceiling = popt[2] >= ceiling_threshold
            return {
                "A": float(popt[0]),
                "n": float(popt[1]),
                "tau": float(popt[2]),
                "C": float(popt[3]),
                "tau_fixed": False,
                "tau_at_ceiling": at_ceiling,
                "r2": float(r2),
                "y_pred": y_pred,
            }
        except Exception:
            return None


def find_residual_peaks(residual, t_s, min_prominence_frac=0.15):
    """Find peaks in the residual that represent ERP components.

    Returns a list of dicts with ``center_s, center_ms, amplitude, width_s,
    polarity, prominence``. Used inside the iterative loop to subtract
    component bumps before re-fitting the smooth arc.
    """
    components = []
    abs_max = np.max(np.abs(residual))
    if abs_max == 0:
        return components
    min_prominence = abs_max * min_prominence_frac

    for polarity, signal in [("positive", residual), ("negative", -residual)]:
        peaks, props = find_peaks(
            signal,
            prominence=min_prominence,
            distance=max(int(len(signal) * 0.05), 1),
            width=3,
        )
        for i, pk in enumerate(peaks):
            amp = residual[pk]  # keep original sign
            center = t_s[pk]

            # Estimate width from half-prominence boundaries
            left_ips = props.get("left_ips", [])
            right_ips = props.get("right_ips", [])
            if len(left_ips) > i and len(right_ips) > i:
                left_idx = max(0, int(left_ips[i]))
                right_idx = min(len(t_s) - 1, int(right_ips[i]))
                width = (t_s[right_idx] - t_s[left_idx]) / 2.355  # FWHM->sigma
            else:
                width = 0.03  # 30 ms default

            components.append(
                {
                    "center_s": float(center),
                    "center_ms": float(center * 1000),
                    "amplitude": float(amp),
                    "width_s": float(max(width, 0.01)),
                    "polarity": polarity,
                    "prominence": float(props["prominences"][i]),
                }
            )

    components.sort(key=lambda c: abs(c["amplitude"]), reverse=True)
    return components


def build_component_waveform(components, t_s):
    """Sum all Gaussian component peaks into a single waveform."""
    total = np.zeros_like(t_s)
    for comp in components:
        total += gaussian_peak(
            t_s,
            comp["amplitude"],
            comp["center_s"],
            comp["width_s"],
        )
    return total


def iterative_decompose(t_s, gfp, max_iter=5, convergence_tol=0.005, fixed_tau=None):
    """Iteratively decompose GFP into a structural arc + component peaks.

    1. Fit gamma+floor to (possibly peak-subtracted) GFP
    2. Find peaks in residual vs. original GFP
    3. Subtract peaks from GFP and re-fit
    4. Repeat until tau stabilizes (< *convergence_tol* seconds)

    Returns the final iteration's results or ``None`` on complete failure.
    """
    peak_idx = int(np.argmax(gfp))
    peak_amp = float(gfp[peak_idx])
    peak_time = max(float(t_s[peak_idx]), 0.02)

    current_gfp = gfp.copy()
    prev_tau = None
    last_good = None

    for _iteration in range(max_iter):
        gamma_fit = fit_gamma_arc(
            t_s,
            current_gfp,
            peak_amp,
            peak_time,
            fixed_tau=fixed_tau,
        )
        if gamma_fit is None:
            break

        # Residual against ORIGINAL GFP
        residual = gfp - gamma_fit["y_pred"]
        components = find_residual_peaks(residual, t_s)
        comp_waveform = build_component_waveform(components, t_s)

        combined = gamma_fit["y_pred"] + comp_waveform
        combined_r2 = float(compute_r_squared(gfp, combined))

        last_good = {
            "gamma": gamma_fit,
            "components": components,
            "comp_waveform": comp_waveform,
            "combined_r2": combined_r2,
            "residual": gfp - combined,
        }

        # Convergence check
        if prev_tau is not None and abs(gamma_fit["tau"] - prev_tau) < convergence_tol:
            break
        prev_tau = gamma_fit["tau"]

        # Subtract components and re-fit
        current_gfp = gfp - comp_waveform

    return last_good
