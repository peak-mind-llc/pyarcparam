"""GFP is the spatial standard deviation across channels, in microvolts."""

import numpy as np

from pyarcparam import compute_gfp

from ._fixtures import gamma_signal, make_synthetic_evoked


def test_gfp_equals_std_across_channels():
    evoked = make_synthetic_evoked(signal_func=gamma_signal(), rng=np.random.default_rng(1))
    times_ms, gfp_uv = compute_gfp(evoked)

    expected = np.std(evoked.data, axis=0) * 1e6
    np.testing.assert_allclose(gfp_uv, expected)
    assert times_ms.shape == gfp_uv.shape
    # Time axis is in ms and spans the epoch.
    np.testing.assert_allclose(times_ms, evoked.times * 1000.0)


def test_gfp_rises_post_stimulus_for_injected_signal():
    evoked = make_synthetic_evoked(signal_func=gamma_signal(), rng=np.random.default_rng(2))
    times_ms, gfp_uv = compute_gfp(evoked)

    pre = gfp_uv[times_ms < 0]
    post_peak = gfp_uv[times_ms >= 0].max()
    # The injected gamma arc makes post-stimulus GFP clearly exceed baseline.
    assert post_peak > pre.max() * 3
