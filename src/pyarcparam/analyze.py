"""Top-level ArcParam analysis: evoked in, descriptors out.

``analyze_evoked`` orchestrates the whole reading:

1. Compute the GFP envelope.
2. **Path B** — fit the gamma descriptor arc (iterative decompose) and read the
   arc anatomy. Every per-peak descriptor rides on this arc.
3. Detect residual peaks and describe each arc-relatively.
4. **Path A** — fit all three models, select by AIC, and report ``best_model``
   + ``tci`` *separately*.
5. Attach traditional labels (annotation only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arc import ArcFit, arc_decay_constant_ms, compute_arc_phases
from .decompose import iterative_decompose
from .fitting import fit_tci
from .gfp import compute_gfp
from .labels import annotate_peaks
from .peaks import PeakDescriptor, detect_residual_peaks

__all__ = ["analyze_evoked", "ArcParamResult"]

# Gamma fit bounds (mirror the descriptor-arc fit in decompose.fit_gamma_arc),
# used to flag degenerate fits parked at a parameter limit.
_GAMMA_N_BOUND_MAX = 20.0
_GAMMA_N_BOUND_MIN = 1.0
_GAMMA_TAU_BOUND_MIN_S = 0.005


@dataclass
class ArcParamResult:
    """Complete ArcParam reading of one evoked response."""

    arc: ArcFit
    peaks: list[PeakDescriptor]
    times_ms: np.ndarray
    gfp_uv: np.ndarray
    residual_uv: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict (camelCase keys matching the ArcParam framework)."""
        return {
            "arc": self.arc.to_dict(),
            "peaks": [p.to_dict() for p in self.peaks],
        }


def analyze_evoked(evoked) -> ArcParamResult | None:
    """Run the full ArcParam analysis on an ``mne.Evoked``.

    Returns ``None`` if the post-stimulus window is too short to fit an arc.
    """
    times_ms, gfp_uv = compute_gfp(evoked)
    times_s = np.asarray(evoked.times)

    post_mask = times_s >= 0
    if not np.any(post_mask):
        return None
    t_post_s = times_s[post_mask].copy()
    gfp_post = gfp_uv[post_mask].copy()
    times_post_ms = t_post_s * 1000.0
    if len(t_post_s) < 10:
        return None

    # Avoid the t=0 singularity for the gamma fit
    dt = float(t_post_s[1] - t_post_s[0]) if len(t_post_s) > 1 else 0.004
    t_fit = t_post_s.copy()
    if t_fit[0] <= 0:
        t_fit[0] = dt * 0.5

    # --- Path B: gamma descriptor arc ---
    decomp = iterative_decompose(t_fit, gfp_post)
    if decomp is None:
        return None
    gamma_fit = decomp["gamma"]
    arc_y = np.asarray(gamma_fit["y_pred"])
    arc_peak_idx = int(np.argmax(arc_y))
    arc_peak_ms = float(times_post_ms[arc_peak_idx])
    arc_peak_amp = float(arc_y[arc_peak_idx])
    arc_decay_ms = arc_decay_constant_ms(arc_y, times_post_ms) or 1.0

    # Degeneracy flags: optimizer parked at a parameter limit -> misleading R^2
    fit_n = float(gamma_fit["n"])
    fit_tau_s = float(gamma_fit["tau"])
    fit_at_bounds: list[str] = []
    if fit_n >= _GAMMA_N_BOUND_MAX - 0.01:
        fit_at_bounds.append("n_max")
    if fit_n <= _GAMMA_N_BOUND_MIN + 0.01:
        fit_at_bounds.append("n_min")
    if fit_tau_s <= _GAMMA_TAU_BOUND_MIN_S + 0.0005:
        fit_at_bounds.append("tau_min")
    if gamma_fit.get("tau_at_ceiling"):
        fit_at_bounds.append("tau_max")
    fit_degenerate = bool(fit_at_bounds)

    # --- Path A: 3-model AIC selection (best_model + canonical TCI) ---
    tci_result = fit_tci(evoked)

    arc = ArcFit(
        arc_peak_ms=arc_peak_ms,
        arc_peak_amplitude_uv=arc_peak_amp,
        decay_const_ms=float(arc_decay_ms),
        tau_ms=float(fit_tau_s * 1000.0),
        n=fit_n,
        floor_uv=float(gamma_fit.get("C", 0.0)),
        gamma_r2=float(gamma_fit.get("r2", 0.0)),
        fit_at_bounds=fit_degenerate,
        fit_at_bounds_params=fit_at_bounds,
        arc_phases=compute_arc_phases(fit_n, fit_tau_s * 1000.0, arc_peak_ms),
        best_model=tci_result.get("best_model"),
        tci=tci_result.get("tci"),
        model_fits=tci_result.get("model_fits", {}),
        arc_times_ms=times_post_ms.tolist(),
        arc_waveform_uv=arc_y.tolist(),
    )

    # --- Residual peaks + descriptors (against the Path B gamma arc) ---
    peaks = detect_residual_peaks(
        gfp_post,
        arc_y,
        times_post_ms,
        arc_peak_ms,
        arc_peak_amp,
        float(arc_decay_ms),
    )
    annotate_peaks(peaks)

    return ArcParamResult(
        arc=arc,
        peaks=peaks,
        times_ms=times_ms,
        gfp_uv=gfp_uv,
        residual_uv=gfp_post - arc_y,
    )
