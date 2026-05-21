"""Arc anatomy — the six numbers that describe a cascade as a whole.

The arc is the smooth envelope the brain's evoked cascade traces out. From the
gamma fit (Path B) we read its shape; the 3-model selection (Path A) tells us
which mathematical species best describes it. Every per-peak descriptor is
expressed relative to this arc.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ArcFit", "arc_decay_constant_ms", "compute_arc_phases"]


@dataclass
class ArcFit:
    """Arc-level descriptors for one evoked response.

    Anatomy fields (``arc_peak_ms``, ``arc_peak_amplitude_uv``,
    ``decay_const_ms``, ``tau_ms``, ``n``, ``floor_uv``, ``gamma_r2``) come
    from the **gamma** descriptor arc (Path B). Model-selection fields
    (``best_model``, ``tci``) come from the **3-model AIC** fit (Path A).

    On a recording where ``best_model`` is ``"log_normal"``, the anatomy is
    *still* gamma-derived — this mirrors the shipping product.
    """

    # --- Path B: gamma descriptor arc ---
    arc_peak_ms: float
    arc_peak_amplitude_uv: float
    decay_const_ms: float
    tau_ms: float
    n: float
    floor_uv: float
    gamma_r2: float
    fit_at_bounds: bool
    fit_at_bounds_params: list[str] = field(default_factory=list)
    arc_phases: dict[str, Any] = field(default_factory=dict)

    # --- Path A: 3-model AIC selection ---
    best_model: str | None = None
    tci: float | None = None
    model_fits: dict[str, Any] = field(default_factory=dict)

    # --- arc waveform (post-stimulus), for plotting ---
    arc_times_ms: list[float] = field(default_factory=list)
    arc_waveform_uv: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-able dict with camelCase keys matching the ArcParam framework."""
        return {
            "arcPeakMs": self.arc_peak_ms,
            "arcPeakAmplitudeUv": self.arc_peak_amplitude_uv,
            "decayConstMs": self.decay_const_ms,
            "tauMs": self.tau_ms,
            "n": self.n,
            "floorUv": self.floor_uv,
            "gammaR2": self.gamma_r2,
            "fitAtBounds": self.fit_at_bounds,
            "fitAtBoundsParams": self.fit_at_bounds_params,
            "bestModel": self.best_model,
            "tci": self.tci,
            "modelFits": self.model_fits,
            "arcPhases": self.arc_phases,
        }


def arc_decay_constant_ms(arc_y: np.ndarray, arc_t_ms: np.ndarray) -> float | None:
    """Time from arc peak to half-amplitude on the decay side (ms).

    Robust across gamma/alpha/log-normal because it reads the fitted arc
    waveform directly. Returns ``None`` if the arc never decays to half.
    """
    if len(arc_y) == 0 or arc_y.max() <= 0:
        return None
    peak_idx = int(np.argmax(arc_y))
    peak_amp = float(arc_y[peak_idx])
    half = peak_amp / 2.0
    for i in range(peak_idx + 1, len(arc_y)):
        if arc_y[i] <= half:
            denom = arc_y[i - 1] - arc_y[i]
            if denom == 0:
                return float(arc_t_ms[i] - arc_t_ms[peak_idx])
            frac = (arc_y[i - 1] - half) / denom
            t_half = arc_t_ms[i - 1] + frac * (arc_t_ms[i] - arc_t_ms[i - 1])
            return float(t_half - arc_t_ms[peak_idx])
    # Never decayed to half within window — fall back to remaining duration
    return float(arc_t_ms[-1] - arc_t_ms[peak_idx])


def compute_arc_phases(n: float, tau_ms: float, arc_peak_ms: float) -> dict[str, Any]:
    """Derive the four natural response phases from gamma (n, tau).

    The gamma function ``f(t) = t^(n-1) * exp(-t/tau)`` has inflection points
    at ``t_peak +/- sqrt(n-1) * tau``. These are the only non-arbitrary time
    landmarks the curve provides, and they partition the response into four
    phases that adapt to the subject's own cascade dynamics:

        MOBILIZING  -> before the rising inflection (steepest acceleration)
        PEAK        -> rising inflection to arc peak (approaching maximum)
        RETURNING   -> arc peak to falling inflection (steepest deceleration)
        SETTLING    -> after the falling inflection (return to baseline)
    """
    sqrt_nm1 = math.sqrt(max(0, n - 1))
    offset = sqrt_nm1 * tau_ms

    rising_ms = max(0.0, arc_peak_ms - offset)
    falling_ms = arc_peak_ms + offset

    phases = [
        {"key": "mobilizing", "label": "Mobilizing", "tmin_ms": 0.0, "tmax_ms": rising_ms},
        {"key": "peak", "label": "Peak", "tmin_ms": rising_ms, "tmax_ms": arc_peak_ms},
        {"key": "returning", "label": "Returning", "tmin_ms": arc_peak_ms, "tmax_ms": falling_ms},
        {"key": "settling", "label": "Settling", "tmin_ms": falling_ms, "tmax_ms": 999.0},
    ]
    return {
        "phases": phases,
        "rising_inflection_ms": rising_ms,
        "falling_inflection_ms": falling_ms,
    }
