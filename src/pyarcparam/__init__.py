"""pyarcparam — a shape-first reference implementation for ERP interpretation.

ArcParam ("Reading the Response Arc") fits a parametric arc to the Global Field
Power envelope of an evoked response, then reads the discrete events that ride
on it as residual peaks, each described in the subject's own arc-relative units.

Quickstart
----------
>>> from pyarcparam import analyze_evoked, load_evoked, diagnostic_figure
>>> evoked = load_evoked("subject_target-ave.fif")   # or pass an mne.Evoked
>>> result = analyze_evoked(evoked)
>>> result.arc.best_model, round(result.arc.tci, 2)
>>> for peak in result.peaks:
...     print(peak.label, peak.latency_ms, peak.boost_ratio)
>>> fig = diagnostic_figure(result); fig.savefig("arc.png")

This is a standalone reference implementation of the method, independent of the
Coherence Workstation product in which it also ships.
"""

from __future__ import annotations

from .analyze import ArcParamResult, analyze_evoked
from .arc import ArcFit, arc_decay_constant_ms, compute_arc_phases
from .decompose import iterative_decompose
from .fitting import fit_all_models, fit_tci
from .gfp import compute_gfp
from .io import load_evoked
from .labels import annotate_peaks, label_for_latency
from .peaks import PeakDescriptor, detect_residual_peaks
from .plotting import diagnostic_figure

__version__ = "0.1.0"

__all__ = [
    "analyze_evoked",
    "ArcParamResult",
    "ArcFit",
    "PeakDescriptor",
    "compute_gfp",
    "iterative_decompose",
    "fit_tci",
    "fit_all_models",
    "detect_residual_peaks",
    "arc_decay_constant_ms",
    "compute_arc_phases",
    "annotate_peaks",
    "label_for_latency",
    "load_evoked",
    "diagnostic_figure",
    "__version__",
]
