"""Traditional component labels — annotation, not analysis.

The traditional names (N1, P2, N2, P3a, P3b) are applied *after* detection by
latency window. They are a convenience for readers who think in component
language; the arc-relative descriptors are the actual analysis. A peak with a
normal-amplitude P3b but ``boost_ratio ~ 0`` is the arc passing through, not a
real allocation event — the label alone would hide that.

This is a deliberately simplified, paradigm-agnostic window table. Coherence
Workstation's shipping detector additionally rescales windows to each
subject's arc peak and validates labels against scalp topography/polarity;
that machinery is out of scope for this reference (see the README).
"""

from __future__ import annotations

from .peaks import PeakDescriptor

__all__ = ["LABEL_WINDOWS", "annotate_peaks", "label_for_latency"]

# (lo_ms, hi_ms, label) — canonical generic windows.
# Latency ranges follow Coherence Workstation's oddball candidate windows.
LABEL_WINDOWS: list[tuple[float, float, str]] = [
    (80.0, 150.0, "N1"),
    (140.0, 210.0, "P2"),
    (210.0, 280.0, "N2"),
    (310.0, 390.0, "P3a"),
    (380.0, 460.0, "P3b"),
]

# Minimum residual amplitude (uV above arc) for a peak to receive any label.
_LABEL_AMP_FLOOR_UV = 0.15


def label_for_latency(latency_ms: float) -> str | None:
    """Return the traditional label whose window center is closest to *latency_ms*.

    Considers every window that contains the latency and returns the label with
    the nearest center, or ``None`` if no window contains it.
    """
    candidates = [
        (abs(latency_ms - (lo + hi) / 2.0), label)
        for lo, hi, label in LABEL_WINDOWS
        if lo <= latency_ms <= hi
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def annotate_peaks(peaks: list[PeakDescriptor]) -> list[PeakDescriptor]:
    """Assign traditional labels to peaks in place (and return the list).

    A peak is labeled only if its positive residual clears the amplitude floor
    and its latency falls in a known window.
    """
    for peak in peaks:
        if peak.residual_amp_uv >= _LABEL_AMP_FLOOR_UV:
            peak.label = label_for_latency(peak.latency_ms)
    return peaks
