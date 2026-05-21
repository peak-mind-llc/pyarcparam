"""Residual peak detection and arc-relative descriptors."""

import numpy as np

from pyarcparam import analyze_evoked, detect_residual_peaks
from pyarcparam.models import gamma_func

from ._fixtures import gamma_with_bump_signal, make_synthetic_evoked


def test_injected_residual_bump_is_detected():
    evoked = make_synthetic_evoked(
        signal_func=gamma_with_bump_signal(bump_center_s=0.18, bump_amp=3.0),
        noise_level=0.05,
        rng=np.random.default_rng(7),
    )
    result = analyze_evoked(evoked)
    assert result is not None
    assert result.peaks, "expected at least one residual peak"

    latencies = [p.latency_ms for p in result.peaks]
    # A detected peak should sit near the injected bump (~180 ms).
    assert min(abs(np.array(latencies) - 180.0)) < 40.0


def test_descriptor_ranges_are_sane():
    evoked = make_synthetic_evoked(
        signal_func=gamma_with_bump_signal(bump_center_s=0.30, bump_amp=2.5),
        noise_level=0.05,
        rng=np.random.default_rng(11),
    )
    result = analyze_evoked(evoked)
    assert result is not None

    for peak in result.peaks:
        # shape_coherence is a product of three [0,1] factors.
        assert 0.0 <= peak.shape_coherence <= 1.0
        if peak.engagement is not None:
            assert peak.engagement >= 0.0
        # A real boost over the envelope is positive.
        if peak.boost_ratio is not None:
            assert peak.boost_ratio > -1.0


def test_pre_mobilization_guard_blanks_boost_ratio():
    """A bump fired before the cascade mobilizes gets no boost_ratio."""
    # Place a bump very early (30 ms) where the gamma carrier is ~0.
    sig = gamma_with_bump_signal(
        A=8.0, n=6.0, tau=0.09, bump_amp=2.0, bump_center_s=0.03, bump_width_s=0.012
    )
    evoked = make_synthetic_evoked(signal_func=sig, noise_level=0.02, rng=np.random.default_rng(3))
    result = analyze_evoked(evoked)
    assert result is not None

    early = [p for p in result.peaks if p.latency_ms < 80.0]
    assert early, "expected an early pre-mobilization peak"
    assert any(p.is_pre_mobilization and p.boost_ratio is None for p in early)


def test_flat_residual_yields_no_peaks():
    # Pure gamma arc, no bump, no noise -> residual is essentially flat.
    t = np.linspace(0.004, 0.8, 200)
    arc_y = gamma_func(t, 8.0, 4.0, 0.08)
    peaks = detect_residual_peaks(
        gfp_post_uv=arc_y,
        arc_y=arc_y,
        times_post_ms=t * 1000.0,
        arc_peak_ms=float(t[np.argmax(arc_y)] * 1000.0),
        arc_peak_amp_uv=float(arc_y.max()),
        arc_decay_ms=80.0,
    )
    assert peaks == []
