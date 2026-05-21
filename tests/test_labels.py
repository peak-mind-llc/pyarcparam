"""Traditional-label annotation (latency windows)."""

from pyarcparam import annotate_peaks, label_for_latency
from pyarcparam.peaks import PeakDescriptor


def test_label_for_latency_windows():
    assert label_for_latency(110) == "N1"
    assert label_for_latency(175) == "P2"
    assert label_for_latency(250) == "N2"
    assert label_for_latency(350) == "P3a"
    assert label_for_latency(420) == "P3b"


def test_label_for_latency_outside_windows():
    assert label_for_latency(50) is None
    assert label_for_latency(600) is None


def test_label_for_latency_picks_nearest_center_on_overlap():
    # 145 ms is in both N1 (80-150) and P2 (140-210); P2's center (175) is
    # farther than N1's (115), so N1 wins.
    assert label_for_latency(145) == "N1"


def _peak(latency_ms, residual_amp_uv):
    return PeakDescriptor(
        latency_ms=latency_ms,
        residual_amp_uv=residual_amp_uv,
        prominence_uv=0.0,
        fwhm_ms=None,
        arc_phase=None,
        arc_relative_amp=None,
        arc_relative_width=None,
        arc_carrier_uv=1.0,
        engagement=0.5,
        boost_ratio=0.2,
        is_pre_mobilization=False,
        shape_coherence=1.0,
        shape_symmetry=1.0,
        shape_unimodality=1.0,
    )


def test_annotate_peaks_respects_amplitude_floor():
    peaks = [_peak(420, residual_amp_uv=0.5), _peak(420, residual_amp_uv=0.05)]
    annotate_peaks(peaks)
    assert peaks[0].label == "P3b"  # above floor
    assert peaks[1].label is None  # below 0.15 uV floor
