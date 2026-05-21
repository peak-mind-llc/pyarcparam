"""Synthetic ERP fixtures for deterministic, offline tests.

Builds ``mne.Evoked`` objects whose GFP traces a known arc shape, optionally
with residual bumps injected at known latencies. No real subject data — every
fixture is generated from a seed so tests are reproducible and PHI-free.

Adapted from Coherence Workstation's ``test_self_coherence`` synthetic-evoked
helper.
"""

from __future__ import annotations

import mne
import numpy as np

from pyarcparam.models import alpha_func, gamma_func, gaussian_peak, lognormal_func

CH_NAMES_19 = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "Fz",
    "Cz",
    "Pz",
]


def make_synthetic_evoked(
    ch_names=None,
    sfreq=256,
    tmin=-0.2,
    tmax=1.0,
    signal_func=None,
    noise_level=0.0,
    rng=None,
):
    """Create a synthetic ``mne.Evoked`` with an optional injected signal.

    ``signal_func`` is evaluated on the post-stimulus time axis (seconds) and
    added to each channel with a random positive spatial weight, so the GFP
    (std across channels) carries the temporal shape.
    """
    if ch_names is None:
        ch_names = CH_NAMES_19
    if rng is None:
        rng = np.random.default_rng(42)

    n_times = int((tmax - tmin) * sfreq)
    n_channels = len(ch_names)
    times = np.linspace(tmin, tmax, n_times)

    data = np.zeros((n_channels, n_times))
    if noise_level > 0:
        data += rng.standard_normal((n_channels, n_times)) * noise_level

    if signal_func is not None:
        post_mask = times >= 0
        t_post = times[post_mask]
        signal = signal_func(t_post)
        weights = rng.uniform(0.2, 2.0, size=n_channels)
        for i in range(n_channels):
            data[i, post_mask] += weights[i] * signal

    data *= 1e-6  # uV -> V (MNE convention)

    info = mne.create_info(ch_names=list(ch_names), sfreq=sfreq, ch_types="eeg")
    return mne.EvokedArray(data, info, tmin=tmin, verbose=False)


def gamma_signal(A=8.0, n=4.0, tau=0.08):
    """A known gamma arc as a signal function of time (s)."""
    return lambda t: gamma_func(t, A, n, tau)


def alpha_signal(A=5.0, tau=0.15):
    return lambda t: alpha_func(t, A, tau)


_LOGNORMAL_MU = float(np.log(0.25))  # peak around 0.25 s


def lognormal_signal(A=2.0, mu=_LOGNORMAL_MU, sigma=0.7):
    """A heavy-tailed log-normal arc (parallel-overlapping shape).

    sigma=0.7 is skewed enough that gamma cannot match it, so the 3-model
    AIC selection reliably prefers log-normal.
    """
    return lambda t: lognormal_func(np.maximum(t, 1e-6), A, mu, sigma)


def gamma_with_bump_signal(
    A=8.0, n=4.0, tau=0.08, bump_amp=3.0, bump_center_s=0.18, bump_width_s=0.02
):
    """A gamma arc plus a Gaussian residual bump at a known latency."""

    def _sig(t):
        return gamma_func(t, A, n, tau) + gaussian_peak(t, bump_amp, bump_center_s, bump_width_s)

    return _sig
