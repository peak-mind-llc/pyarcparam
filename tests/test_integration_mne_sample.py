"""Integration test on MNE's built-in sample dataset.

Proves ArcParam runs end-to-end on a real evoked response from a standard MNE
dataset (no synthetic data). Network-gated: the dataset downloads on first run.
Run with ``pytest -m network``; skipped by default.
"""

import pytest

from pyarcparam import analyze_evoked


@pytest.mark.network
def test_analyze_mne_sample_auditory_evoked():
    mne = pytest.importorskip("mne")
    from mne.datasets import sample

    data_path = sample.data_path(download=True)
    fname = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"

    evoked = mne.read_evokeds(fname, condition="Left Auditory", verbose="ERROR")
    evoked.pick("eeg")

    result = analyze_evoked(evoked)
    assert result is not None
    assert result.arc.arc_peak_ms > 0
    assert result.arc.best_model in ("alpha", "gamma", "log_normal")
    # The auditory evoked has clear components -> at least one residual peak.
    assert len(result.peaks) >= 1
