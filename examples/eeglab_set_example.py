"""Reading ArcParam from an EEGLAB ``.set`` dataset.

ArcParam works on an averaged evoked response. EEGLAB epoched datasets reach
that via MNE's EEGLAB reader. ``load_evoked`` does this for you:

    python examples/eeglab_set_example.py path/to/subject_epochs.set

If you have epochs in memory already, the two lines that matter are::

    epochs = mne.io.read_epochs_eeglab("subject_epochs.set")
    evoked = epochs["target"].average()   # pick your condition, then average
    result = analyze_evoked(evoked)

For condition contrasts (e.g. an oddball Go vs. NoGo), analyze each condition's
average separately and compare the arcs and per-peak descriptors.
"""

from __future__ import annotations

import sys

from pyarcparam import analyze_evoked, load_evoked


def main(argv: list[str]) -> None:
    if len(argv) < 2:
        print(__doc__)
        print("Usage: python examples/eeglab_set_example.py path/to/epochs.set")
        return

    evoked = load_evoked(argv[1])  # EEGLAB .set epochs -> averaged evoked
    print(f"Loaded evoked: {len(evoked.ch_names)} ch, {evoked.times.size} samples")

    result = analyze_evoked(evoked)
    if result is None:
        print("Could not fit an arc.")
        return

    print(f"best_model={result.arc.best_model}  TCI={result.arc.tci}")
    for p in result.peaks:
        print(f"  {p.label or '?'} @ {p.latency_ms:.0f} ms  boostRatio={p.boost_ratio}")


if __name__ == "__main__":
    main(sys.argv)
