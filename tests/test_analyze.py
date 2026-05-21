"""End-to-end orchestration: evoked -> ArcParamResult."""

import numpy as np

from pyarcparam import analyze_evoked, diagnostic_figure

from ._fixtures import gamma_with_bump_signal, make_synthetic_evoked


def _result(seed=5):
    evoked = make_synthetic_evoked(
        signal_func=gamma_with_bump_signal(bump_center_s=0.30, bump_amp=2.5),
        noise_level=0.05,
        rng=np.random.default_rng(seed),
    )
    return analyze_evoked(evoked)


def test_analyze_returns_full_result():
    result = _result()
    assert result is not None

    arc = result.arc
    # Path B (gamma) anatomy.
    assert arc.arc_peak_ms > 0
    assert arc.arc_peak_amplitude_uv > 0
    assert arc.decay_const_ms > 0
    assert arc.tau_ms > 0
    assert arc.n >= 1.0
    assert "phases" in arc.arc_phases

    # Path A (3-model AIC) selection reported separately.
    assert arc.best_model in ("alpha", "gamma", "log_normal")
    assert arc.tci is not None and 0.0 <= arc.tci <= 1.0


def test_to_dict_uses_camelcase_framework_keys():
    result = _result()
    d = result.to_dict()
    expected_arc_keys = {
        "arcPeakMs",
        "arcPeakAmplitudeUv",
        "decayConstMs",
        "tauMs",
        "tci",
        "bestModel",
    }
    assert expected_arc_keys <= set(d["arc"].keys())
    if d["peaks"]:
        peak_keys = set(d["peaks"][0].keys())
        assert {"arcPhase", "engagement", "boostRatio", "shapeCoherence", "label"} <= peak_keys


def test_short_post_window_returns_none():
    # Almost no post-stimulus samples -> cannot fit an arc.
    evoked = make_synthetic_evoked(tmin=-0.2, tmax=0.02, sfreq=256)
    assert analyze_evoked(evoked) is None


def test_diagnostic_figure_builds():
    import matplotlib

    matplotlib.use("Agg")
    result = _result()
    fig = diagnostic_figure(result)
    assert fig is not None
    assert len(fig.axes) == 2
