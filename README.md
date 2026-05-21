# pyarcparam

**A shape-first reference implementation for ERP interpretation — "Reading the Response Arc."**

ArcParam reads an evoked response not as a sum of labeled components, but as a
whole shape. It fits a parametric arc to the **Global Field Power (GFP)
envelope**, then reads the discrete events that ride on that arc as residual
peaks — each described in the subject's *own* arc-relative units.

This repository is a small, standalone, dependency-light reference
implementation of the core method. It runs on standard
[MNE-Python](https://mne.tools) / EEGLAB evoked data. It is intentionally
independent of the Coherence Workstation product in which the method also
ships.

> **Not a clinical tool.** ArcParam describes the *shape* of a response — what
> the cascade is doing — not a diagnosis. Population norms, validation, and
> clinical interpretation live elsewhere. Use the descriptors as questions to
> ask, not conclusions to draw.

---

## Install

```bash
pip install -e ".[dev]"     # from a clone
```

Requires Python ≥ 3.10. Runtime deps: `numpy`, `scipy`, `mne`, `matplotlib`.

## Quickstart

```python
from pyarcparam import analyze_evoked, load_evoked, diagnostic_figure

evoked = load_evoked("subject_target-ave.fif")   # or an EEGLAB .set, or an mne.Evoked
result = analyze_evoked(evoked)

print(result.arc.best_model, round(result.arc.tci, 2))     # e.g. "gamma" 0.97
for peak in result.peaks:
    print(peak.label, round(peak.latency_ms), peak.boost_ratio)

fig = diagnostic_figure(result)
fig.savefig("arc.png")
```

No data on hand? `python examples/quickstart.py` runs on a synthetic evoked and
writes a diagnostic figure.

---

## The method

### 1. GFP — one curve for the whole scalp

GFP is the spatial standard deviation across all electrodes at each time point.
When the brain produces a synchronized response, electrodes deviate together
and GFP rises. The result is a single curve: how much the whole scalp is doing
at each moment. It rises, peaks, and falls — one "mountain," the evoked cascade
considered as a single event.

### 2. Fit the arc

We fit a parametric impulse-response curve to the GFP envelope and read the
cascade's anatomy from it:

| descriptor | meaning |
|---|---|
| `arcPeakMs` | when the cascade peaks |
| `arcPeakAmplitudeUv` | total response magnitude |
| `decayConstMs` | peak to half-height on the way down |
| `tau` (`tauMs`) | decay tempo — how quickly the cascade releases |
| `n` | gamma shape parameter — effectively the cascade's depth |
| `TCI` | **Temporal Coherence Index** — R² of the best-fit arc |

**TCI** measures how well a single smooth curve explains the actual GFP. Near 1,
the cascade unfolds as one coherent event; below ~0.7 it fragments into
multiple sub-events the smooth arc can't capture. (Most healthy recordings sit
above ~0.85; the clinically interesting low-TCI cases are rare.)

### 3. Read the residual

The fitted arc captures the smooth envelope. What's left over — the
**residual** — captures the discrete events riding on it. We detect peaks in
the residual, not in fixed component windows, and describe each one relative to
*this subject's* arc:

| descriptor | meaning |
|---|---|
| `arcPhase` | `(peakLatency − arcPeakMs) / decayConstMs` — negative = mobilizing, 0 = apex, positive = settling |
| `engagement` | arc envelope height at the peak's latency, normalized to arc peak (1 = fully mobilized) |
| `boostRatio` | how much the peak does *above* the arc envelope (0 = the arc just passing through) |
| `shapeCoherence` | symmetry × unimodality × smoothness (0–1) — how cleanly the peak behaves as one event |

The traditional names (N1, P2, P3a, P3b) are attached afterward by latency
window as **annotation**. A peak with a normal-amplitude P3b but `boostRatio ≈ 0`
is the arc passing through — no specific allocation event — which a label alone
would hide.

---

## Two fits, on purpose

This reference reproduces a subtlety of the shipping product: **it runs the arc
fit two ways.**

- **Path A — model selection (3 candidates, AIC).** We fit *alpha*, *gamma*, and
  *log-normal* impulse responses to the smoothed GFP and pick the best by AIC.
  The winner is `best_model` (the "math species" — sequential vs.
  parallel-overlapping) and its R² is the canonical `tci`.

- **Path B — descriptor arc (gamma only, iterative).** The arc that every
  per-peak descriptor (`arcPhase`, `engagement`, `boostRatio`) and the arc
  anatomy are measured against is fit with **gamma + a floor only**, refined by
  iteratively subtracting detected peaks and re-fitting.

**Consequence:** on a recording where AIC selects `log_normal`, `best_model`
will say `log_normal` while the descriptors are *still* computed against a gamma
arc. That is exactly how the product behaves today. The training narrative
describes a single unified three-model method; this repo matches the
implementation rather than silently reconciling the two. (`ArcFit` exposes both
`tci` from Path A and `gamma_r2` from Path B so the difference is visible.)

---

## API

| symbol | purpose |
|---|---|
| `analyze_evoked(evoked) -> ArcParamResult \| None` | full analysis (GFP → arc → peaks → labels) |
| `load_evoked(path)` | load `.fif` evoked/epochs or EEGLAB `.set` epochs → averaged `mne.Evoked` |
| `diagnostic_figure(result)` | matplotlib figure: GFP + fitted arc, residual + peaks |
| `compute_gfp(evoked)` | `(times_ms, gfp_uv)` |
| `iterative_decompose(t, gfp)` | Path B gamma descriptor arc |
| `fit_tci(evoked)` / `fit_all_models(...)` | Path A 3-model AIC selection |
| `detect_residual_peaks(...)` | residual peaks + descriptors |

`ArcParamResult.to_dict()` returns a JSON-able dict with camelCase keys
(`arcPeakMs`, `boostRatio`, …).

### What's intentionally out of scope

To keep the reference small and free of clinical IP, it omits the product's
SVD-based source attribution, the full arc-scaled label *validation* (topography
and polarity gates), normative-cohort comparison, and any AI-generated
interpretation. Those build on top of the descriptors computed here.

---

## Tests

```bash
pytest -m "not network" -q     # offline, deterministic (synthetic arcs)
pytest -m network -q           # also runs on MNE's sample dataset (downloads)
```

Synthetic fixtures generate known gamma / log-normal arcs (optionally with
injected residual bumps) and assert the fitter recovers them. **No real subject
data is included** — the worked clinical cases discussed in the training
material are described, not shipped.

---

## Roadmap

Design questions, known limitations, and research directions (e.g. treating the
gamma fit as a *carrier* rather than *the model*, and a continuous
departure-from-gamma descriptor) live in [docs/ROADMAP.md](docs/ROADMAP.md).

## License

[Apache-2.0](LICENSE). The ArcParam method was developed by James Croall;
see [NOTICE](NOTICE).
