"""Peaks as residuals — the events riding on the arc.

The arc is the smooth baseline. The actual GFP has bumps above it: moments
where the brain does more than the arc predicts. We detect peaks in the
*residual* (GFP minus the gamma descriptor arc), not in fixed component
windows, and describe each in the subject's own arc-relative units:

* ``arc_phase``      — where the peak sits in the cascade timeline
* ``engagement``     — how mobilized the cascade is at the peak's latency
* ``boost_ratio``    — how much the peak does above the arc envelope
* ``shape_coherence``— how cleanly the peak behaves as a single event

Faithful port of Coherence Workstation's residual-descriptor loop in
``detection_arcparam`` (the SVD source attribution and label-validation gates
are intentionally omitted — see the README).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks, peak_widths

__all__ = [
    "PeakDescriptor",
    "compute_fwhm_lookup",
    "compute_shape_coherence",
    "detect_residual_peaks",
]


@dataclass
class PeakDescriptor:
    """Arc-relative descriptors for one residual peak.

    ``label`` is filled in by :func:`pyarcparam.labels.annotate_peaks` and is
    annotation, not analysis — the four arc-relative numbers are the analysis.
    """

    latency_ms: float
    residual_amp_uv: float
    prominence_uv: float
    fwhm_ms: float | None
    arc_phase: float | None
    arc_relative_amp: float | None
    arc_relative_width: float | None
    arc_carrier_uv: float
    engagement: float | None
    boost_ratio: float | None
    is_pre_mobilization: bool
    shape_coherence: float
    shape_symmetry: float
    shape_unimodality: float
    label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "latencyMs": self.latency_ms,
            "residualAmpUv": self.residual_amp_uv,
            "prominenceUv": self.prominence_uv,
            "fwhmMs": self.fwhm_ms,
            "arcPhase": self.arc_phase,
            "arcRelativeAmp": self.arc_relative_amp,
            "arcRelativeWidth": self.arc_relative_width,
            "arcCarrierUv": self.arc_carrier_uv,
            "engagement": self.engagement,
            "boostRatio": self.boost_ratio,
            "isPreMobilization": self.is_pre_mobilization,
            "shapeCoherence": self.shape_coherence,
            "shapeSymmetry": self.shape_symmetry,
            "shapeUnimodality": self.shape_unimodality,
            "label": self.label,
        }


def compute_fwhm_lookup(
    residual: np.ndarray,
    times_post_ms: np.ndarray,
    peak_indices: np.ndarray,
) -> dict[int, float]:
    """Run ``scipy.signal.peak_widths`` and return ``{rounded_latency_ms: fwhm_ms}``."""
    if len(peak_indices) == 0 or len(residual) == 0:
        return {}
    if len(times_post_ms) > 1:
        per_ms = (len(times_post_ms) - 1) / (float(times_post_ms[-1]) - float(times_post_ms[0]))
    else:
        per_ms = 1.0
    try:
        widths_samples, _, _, _ = peak_widths(residual, peak_indices, rel_height=0.5)
    except Exception:
        return {}
    return {
        int(round(float(times_post_ms[idx]))): float(w_samples / per_ms)
        for idx, w_samples in zip(peak_indices, widths_samples, strict=True)
    }


def compute_shape_coherence(
    residual: np.ndarray,
    peak_idx: int,
    times_post_ms: np.ndarray,
    fwhm_ms: float | None,
) -> dict[str, float]:
    """Measure how much a residual peak looks like a single coherent event.

    Three sub-metrics (each 0-1), multiplied to give ``shape_coherence``:

    1. **Symmetry**: rise time vs. fall time from half-max crossings.
    2. **Unimodality**: ``1 / n_local_maxima`` in the window.
    3. **Smoothness**: ``1 / (1 + n_extra_zero_crossings)`` of the first
       derivative.

    The per-component analog of 1/f spectral coherence: a coherent ERP peak has
    a smooth, symmetric rise-and-fall.
    """
    n = len(residual)
    if n == 0 or peak_idx < 0 or peak_idx >= n:
        return {"shape_coherence": 0.0, "shape_symmetry": 0.0, "shape_unimodality": 0.0}

    if len(times_post_ms) > 1:
        dt = float(times_post_ms[1] - times_post_ms[0])
    else:
        dt = 4.0  # fallback
    half_win_ms = max(50.0, (fwhm_ms or 50.0) * 2.0)
    half_win_idx = max(5, int(half_win_ms / dt))
    lo = max(0, peak_idx - half_win_idx)
    hi = min(n, peak_idx + half_win_idx + 1)
    window = residual[lo:hi]
    pk_in_win = peak_idx - lo

    peak_val = float(residual[peak_idx])
    if peak_val <= 0:
        return {"shape_coherence": 0.0, "shape_symmetry": 0.0, "shape_unimodality": 0.0}

    # Symmetry: half-max rise time vs. fall time
    half_max = peak_val * 0.5
    rise_idx = pk_in_win
    while rise_idx > 0 and window[rise_idx] > half_max:
        rise_idx -= 1
    rise_samples = pk_in_win - rise_idx
    fall_idx = pk_in_win
    while fall_idx < len(window) - 1 and window[fall_idx] > half_max:
        fall_idx += 1
    fall_samples = fall_idx - pk_in_win
    total_hm = rise_samples + fall_samples
    symmetry = 1.0 - abs(rise_samples - fall_samples) / total_hm if total_hm > 0 else 0.0

    # Unimodality: count significant local maxima (above 20% of peak height)
    height_threshold = peak_val * 0.20
    n_maxima = 0
    for i in range(1, len(window) - 1):
        if window[i] > window[i - 1] and window[i] > window[i + 1] and window[i] > height_threshold:
            n_maxima += 1
    unimodality = 1.0 / max(1, n_maxima)

    # Smoothness: significant derivative direction reversals (>10% of peak)
    diff = np.diff(window)
    reversal_threshold = peak_val * 0.10
    sign_changes = 0
    for i in range(1, len(diff)):
        if diff[i - 1] * diff[i] < 0:
            local_swing = abs(window[i + 1] - window[i]) if i + 1 < len(window) else 0
            if local_swing > reversal_threshold:
                sign_changes += 1
    extra_crossings = max(0, sign_changes - 1)
    smoothness = 1.0 / (1.0 + extra_crossings)

    return {
        "shape_coherence": float(symmetry * unimodality * smoothness),
        "shape_symmetry": float(symmetry),
        "shape_unimodality": float(unimodality),
    }


# Below this engagement, a peak is firing before the cascade has meaningfully
# mobilized; boost_ratio against a near-zero carrier is meaningless, so we
# withhold it and flag the peak instead.
_PRE_MOBILIZATION_ENGAGEMENT = 0.05


def detect_residual_peaks(
    gfp_post_uv: np.ndarray,
    arc_y: np.ndarray,
    times_post_ms: np.ndarray,
    arc_peak_ms: float,
    arc_peak_amp_uv: float,
    arc_decay_ms: float,
) -> list[PeakDescriptor]:
    """Detect peaks in the residual (GFP - arc) and describe each arc-relatively.

    Parameters mirror the gamma descriptor arc from
    :func:`pyarcparam.decompose.iterative_decompose`. ``arc_y`` is the
    post-stimulus gamma arc sampled on ``times_post_ms``.
    """
    residual = gfp_post_uv - arc_y
    res_std = float(np.std(residual)) if len(residual) > 1 else 0.0
    if res_std == 0:
        return []

    per_ms = (len(times_post_ms) - 1) / (float(times_post_ms[-1]) - float(times_post_ms[0]))
    min_distance = max(1, int(40.0 * per_ms))  # 40 ms
    peak_indices, peak_props = find_peaks(
        residual,
        prominence=0.4 * res_std,
        distance=min_distance,
    )
    prominences = peak_props.get("prominences", np.array([]))
    fwhm_lookup = compute_fwhm_lookup(residual, times_post_ms, peak_indices)

    peaks: list[PeakDescriptor] = []
    for i, idx in enumerate(peak_indices):
        latency_ms = float(times_post_ms[idx])
        residual_amp = float(residual[idx])
        prominence = float(prominences[i]) if i < len(prominences) else 0.0

        fwhm_ms = fwhm_lookup.get(int(round(latency_ms)))
        arc_relative_width = (
            float(fwhm_ms / arc_decay_ms) if (fwhm_ms is not None and arc_decay_ms > 0) else None
        )
        arc_phase = float((latency_ms - arc_peak_ms) / arc_decay_ms) if arc_decay_ms > 0 else None
        arc_relative_amp = float(residual_amp / arc_peak_amp_uv) if arc_peak_amp_uv > 0 else None

        # Arc carrier (envelope height at this latency) and engagement
        arc_carrier = float(np.interp(latency_ms, times_post_ms, arc_y))
        engagement = float(arc_carrier / arc_peak_amp_uv) if arc_peak_amp_uv > 0 else None

        # Boost ratio with pre-mobilization guard
        if engagement is not None and engagement < _PRE_MOBILIZATION_ENGAGEMENT:
            boost_ratio: float | None = None
            is_pre_mobilization = True
        else:
            boost_ratio = float(residual_amp / arc_carrier) if arc_carrier > 0 else None
            is_pre_mobilization = False

        shape = compute_shape_coherence(residual, int(idx), times_post_ms, fwhm_ms)

        peaks.append(
            PeakDescriptor(
                latency_ms=latency_ms,
                residual_amp_uv=residual_amp,
                prominence_uv=prominence,
                fwhm_ms=fwhm_ms,
                arc_phase=arc_phase,
                arc_relative_amp=arc_relative_amp,
                arc_relative_width=arc_relative_width,
                arc_carrier_uv=arc_carrier,
                engagement=engagement,
                boost_ratio=boost_ratio,
                is_pre_mobilization=is_pre_mobilization,
                shape_coherence=shape["shape_coherence"],
                shape_symmetry=shape["shape_symmetry"],
                shape_unimodality=shape["shape_unimodality"],
            )
        )

    return peaks
