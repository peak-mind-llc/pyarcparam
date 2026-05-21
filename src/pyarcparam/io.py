"""Loading evoked ERP data from MNE and EEGLAB files.

ArcParam operates on an averaged evoked response (``mne.Evoked``). This helper
reaches that common currency from the formats clinicians actually have:

* ``.fif`` — MNE evoked (``*-ave.fif``) or epochs (``*-epo.fif``, averaged here).
* ``.set`` — EEGLAB epoched dataset (averaged here via MNE's EEGLAB reader).

Continuous recordings (raw ``.edf``/``.bdf``/continuous ``.set``) must be
epoched around events before they can be averaged; that is paradigm-specific
and out of scope for this reference.
"""

from __future__ import annotations

from pathlib import Path

import mne

__all__ = ["load_evoked"]


def load_evoked(path, condition: str | int | None = None) -> mne.Evoked:
    """Load an averaged evoked response from a file.

    Parameters
    ----------
    path : str | Path | mne.Evoked
        Path to a ``.fif`` evoked/epochs file or an EEGLAB ``.set`` epoched
        dataset. If an ``mne.Evoked`` is passed, it is returned unchanged.
    condition : str | int | None
        For multi-condition ``-ave.fif`` files, the condition to select
        (passed to ``mne.read_evokeds``). Ignored for epochs/EEGLAB files.

    Returns
    -------
    mne.Evoked
    """
    if isinstance(path, mne.Evoked):
        return path

    p = Path(path)
    suffix = p.suffix.lower()
    name = p.name.lower()

    if suffix == ".fif":
        # Epochs file -> average; otherwise read as evoked.
        if name.endswith("-epo.fif") or name.endswith("_epo.fif"):
            return mne.read_epochs(p, verbose="ERROR").average()
        try:
            evokeds = mne.read_evokeds(p, condition=condition, verbose="ERROR")
        except Exception:
            # Fall back to treating it as epochs.
            return mne.read_epochs(p, verbose="ERROR").average()
        if isinstance(evokeds, list):
            return evokeds[0]
        return evokeds

    if suffix == ".set":
        # EEGLAB epoched dataset -> average.
        epochs = mne.io.read_epochs_eeglab(str(p), verbose="ERROR")
        return epochs.average()

    raise ValueError(
        f"Unsupported file type {suffix!r} for {p}. "
        "ArcParam needs an averaged evoked response. Supported: .fif "
        "(evoked or epochs) and EEGLAB epoched .set. Continuous recordings "
        "(.edf/.bdf/continuous .set) must be epoched and averaged first."
    )
