"""Global Field Power — the substrate ArcParam reads.

GFP is the spatial standard deviation across all electrodes at each time
point. When the brain produces a synchronized response, electrodes deviate
together and GFP rises: one curve describing how much the whole scalp is
doing at each moment. The fitted arc and its residual peaks are read off
this single envelope.
"""

from __future__ import annotations

import numpy as np

__all__ = ["compute_gfp"]


def compute_gfp(evoked):
    """Global Field Power of an evoked response, in microvolts.

    Parameters
    ----------
    evoked : mne.Evoked
        Averaged evoked response. MNE stores data in volts.

    Returns
    -------
    times_ms : np.ndarray
        Time axis in milliseconds (same length as ``evoked.times``).
    gfp_uv : np.ndarray
        GFP = ``std(data, axis=0)`` converted to microvolts.
    """
    data = evoked.data  # (n_channels, n_times), volts
    gfp_uv = np.std(data, axis=0) * 1e6  # V -> uV
    times_ms = np.asarray(evoked.times) * 1000.0
    return times_ms, gfp_uv
