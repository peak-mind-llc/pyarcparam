"""ArcParam quickstart.

Run on a synthetic evoked (no data files needed):

    python examples/quickstart.py

Or point it at your own averaged evoked:

    python examples/quickstart.py path/to/subject-ave.fif
    python examples/quickstart.py path/to/subject_epochs.set
"""

from __future__ import annotations

import sys

import numpy as np

from pyarcparam import analyze_evoked, diagnostic_figure, load_evoked
from pyarcparam.models import gamma_func, gaussian_peak


def _synthetic_evoked():
    """A 19-channel oddball-style evoked: gamma arc + a couple of events."""
    import mne

    rng = np.random.default_rng(42)
    ch_names = [
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
    sfreq, tmin, tmax = 256, -0.2, 1.0
    times = np.linspace(tmin, tmax, int((tmax - tmin) * sfreq))
    post = times >= 0
    t = times[post]

    arc = gamma_func(t, A=8.0, n=4.0, tau=0.10)
    arc += gaussian_peak(t, 3.0, 0.18, 0.02)  # an early event
    arc += gaussian_peak(t, 2.0, 0.40, 0.03)  # a P3b-ish event

    data = rng.standard_normal((len(ch_names), times.size)) * 0.1
    weights = rng.uniform(0.4, 1.6, size=len(ch_names))
    for i in range(len(ch_names)):
        data[i, post] += weights[i] * arc
    data *= 1e-6  # uV -> V

    info = mne.create_info(ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.EvokedArray(data, info, tmin=tmin, verbose=False)


def main(argv: list[str]) -> None:
    evoked = load_evoked(argv[1]) if len(argv) > 1 else _synthetic_evoked()

    result = analyze_evoked(evoked)
    if result is None:
        print("Could not fit an arc (post-stimulus window too short?).")
        return

    arc = result.arc
    print("\n=== Arc (the cascade as a whole) ===")
    print(f"  best_model        : {arc.best_model}")
    print(f"  TCI (best-fit R^2): {arc.tci:.3f}" if arc.tci is not None else "  TCI: n/a")
    print(f"  arcPeakMs         : {arc.arc_peak_ms:.0f} ms")
    print(f"  arcPeakAmplitude  : {arc.arc_peak_amplitude_uv:.2f} uV")
    print(f"  decayConstMs      : {arc.decay_const_ms:.0f} ms")
    print(f"  tau / n           : {arc.tau_ms:.0f} ms / {arc.n:.1f}")

    print("\n=== Peaks (events riding on the arc) ===")
    for p in result.peaks:
        boost = f"{p.boost_ratio:.2f}" if p.boost_ratio is not None else "pre-mob"
        print(
            f"  {(p.label or '?'):>5}  {p.latency_ms:4.0f} ms  "
            f"arcPhase={p.arc_phase:+.2f}  engagement={p.engagement:.2f}  "
            f"boost={boost}  shapeCoherence={p.shape_coherence:.2f}"
        )

    fig = diagnostic_figure(result)
    out = "arc.png"
    fig.savefig(out, dpi=130)
    print(f"\nSaved diagnostic figure to {out}")


if __name__ == "__main__":
    main(sys.argv)
