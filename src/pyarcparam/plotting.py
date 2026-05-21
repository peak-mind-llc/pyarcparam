"""Diagnostic figure — the picture behind the numbers.

``diagnostic_figure`` reproduces the layout of the worked-case figures in the
ArcParam training material:

* GFP envelope with the fitted arc overlaid, peak and decay landmarks marked.
* The residual with detected peaks, in arc-relative units.
* A descriptor table for each detected peak.

No clinical interpretation is drawn — only the computation made visible.
"""

from __future__ import annotations

import numpy as np

from .analyze import ArcParamResult

__all__ = ["diagnostic_figure"]


def diagnostic_figure(result: ArcParamResult, title: str | None = None):
    """Build a matplotlib Figure summarizing an ArcParam result.

    Returns the ``matplotlib.figure.Figure`` (not shown or saved).
    """
    import matplotlib.pyplot as plt

    arc = result.arc
    arc_t = np.asarray(arc.arc_times_ms)
    arc_y = np.asarray(arc.arc_waveform_uv)

    # Post-stimulus GFP, aligned to the arc time axis.
    post = result.times_ms >= 0
    gfp_t = result.times_ms[post]
    gfp_y = result.gfp_uv[post]
    residual = result.residual_uv

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, height_ratios=[2, 1])
    ax_arc, ax_res = axes

    # --- Top: GFP + fitted arc ---
    ax_arc.plot(gfp_t, gfp_y, color="0.4", lw=1.2, label="GFP")
    ax_arc.plot(arc_t, arc_y, color="C0", lw=2.2, label=f"arc ({arc.best_model or 'gamma'})")
    ax_arc.axvline(arc.arc_peak_ms, color="C0", ls="--", lw=1, alpha=0.7)
    ax_arc.annotate(
        f"arcPeak {arc.arc_peak_ms:.0f} ms\n{arc.arc_peak_amplitude_uv:.2f} uV",
        xy=(arc.arc_peak_ms, arc.arc_peak_amplitude_uv),
        xytext=(8, -4),
        textcoords="offset points",
        fontsize=8,
        color="C0",
    )
    tci_txt = f"{arc.tci:.2f}" if arc.tci is not None else "n/a"
    ax_arc.set_title(
        title
        or (
            f"ArcParam — best_model={arc.best_model}  TCI={tci_txt}  "
            f"tau={arc.tau_ms:.0f} ms  n={arc.n:.1f}"
        )
    )
    ax_arc.set_ylabel("GFP (uV)")
    ax_arc.legend(loc="upper right", fontsize=8)
    ax_arc.grid(alpha=0.2)

    # --- Bottom: residual + detected peaks ---
    ax_res.axhline(0, color="0.7", lw=0.8)
    ax_res.plot(arc_t[: len(residual)], residual, color="0.3", lw=1.0, label="residual")
    for peak in result.peaks:
        ax_res.plot(peak.latency_ms, peak.residual_amp_uv, "o", color="C3", ms=5)
        lbl = peak.label or ""
        boost = f"x{peak.boost_ratio:.2f}" if peak.boost_ratio is not None else "pre-mob"
        ax_res.annotate(
            f"{lbl}\n{peak.latency_ms:.0f}ms\nboost {boost}",
            xy=(peak.latency_ms, peak.residual_amp_uv),
            xytext=(0, 8),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            color="C3",
        )
    ax_res.set_ylabel("residual (uV)")
    ax_res.set_xlabel("time (ms)")
    ax_res.grid(alpha=0.2)

    fig.tight_layout()
    return fig
