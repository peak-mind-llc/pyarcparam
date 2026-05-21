# Roadmap & open questions

This is a reference implementation, and the method behind it is still being
developed. This document captures design questions, known limitations, and
research directions — not a committed release plan. Contributions and pushback
welcome.

> Reminder: ArcParam describes the *shape* of a response, not a diagnosis. The
> patterns and hypotheses below are questions to investigate, not conclusions.

---

## Known limitations & design questions

### 1. Two fits under one reading — "carrier" vs. "model"

The implementation fits the arc two ways (see the README's *Two fits, on
purpose*):

- **Path B (gamma, iterative)** — the arc that every per-peak descriptor and the
  arc anatomy are measured against.
- **Path A (3-model AIC)** — selects `best_model` and the canonical `tci`.

These answer genuinely different questions: Path B is a *baseline-subtraction*
question ("what smooth envelope do the events ride on?"), Path A is a
*classification* question ("what shape family is this cascade?"). The open
design question is whether to **stop presenting them as one reading**:

- Reframe the gamma fit as the **carrier** (a deliberately stable, unimodal
  baseline with clean inflection geometry), not "the model." Gamma is a better
  *baseline to subtract* than the AIC winner even when it's a worse *model* — it
  is strictly unimodal, has the closed-form inflection structure `arc_phases`
  depends on, is robust to iterative peak-subtraction, and has no `1/t`
  singularity to distort `engagement`/`boost_ratio` ratios in the tail.
- Report the **species** (gamma / log-normal / alpha) as a separate
  whole-cascade descriptor.
- Keep **one canonical TCI** = best-fit R² (the method's definition), and expose
  the gamma-carrier R² under a distinct name. (`ArcFit` already separates `tci`
  from `gamma_r2` — the remaining work is to make the *naming and framing*
  reflect that these measure different things.)

### 2. `boost_ratio` ↔ carrier-misfit ambiguity

When a non-gamma shape fits the envelope better, the gamma carrier *misfits*,
so the residual (`GFP − carrier`) carries leftover smooth arc-structure in
addition to discrete events. `boost_ratio` then conflates **"a real discrete
event"** with **"the carrier is the wrong shape here."** This ambiguity is worst
exactly when the cascade departs from the organized-sequential form — i.e. the
clinically interesting cases. A descriptor that *separates* these two is needed
before `boost_ratio` can be trusted on non-gamma recordings (see Research
direction C).

---

## Research directions

### A. Departure-from-gamma as a continuous descriptor

Mechanistically, gamma `t^(n-1)·e^(-t/τ)` is the impulse response of *n
sequential stages* (Erlang). It is the **organized-sequential prior**. The model
competition flips to a non-gamma family only when the response *departs* from
that form.

That makes the **gap between the gamma fit and the best non-gamma fit** a
candidate dysregulation metric — "how far has this cascade departed from
organized-sequential." A continuous `ΔAIC` (or `ΔR²`) is more robust than the
categorical `best_model`, which is brittle: near a model boundary, small noise
flips the winner. Proposed: add `delta_aic` / `departure_from_gamma` to
`ArcFit` and report the continuous departure, not just the label.

### B. Two orthogonal axes of disorganization

`tci` and "departure" measure different things and should not be conflated:

- **Low TCI → fragmentation.** No single smooth model fits at all (multiple
  sub-events the arc can't capture).
- **High departure (best_model ≠ gamma) → non-sequentiality.** A single smooth
  model fits fine, but it is the dispersed/shallow family, not the sequential
  one.

This yields a clean 2×2: *coherent-sequential* (healthy gamma) /
*coherent-dispersed* (log-normal wins, high R² for the "wrong" family —
organized but slow to release) / *fragmented* (low TCI). The middle cell is a
real category the current design under-uses.

### C. Residual decomposition

Split the residual into **(smooth low-frequency misfit)** + **(discrete
events)**. The smooth component becomes its own descriptor (a "doesn't release"
/ dysregulation envelope), and `boost_ratio` is computed against the discrete
part only — resolving the ambiguity in limitation #2.

### D. Mechanistic reading of the species

- **gamma** ↔ sequential, n-stage cascade (organized).
- **log-normal** ↔ parallel-overlapping, heavy right tail ("can't release").
- **alpha** ↔ shallow, few-stage, front-loaded (n≈2).

Note the **asymmetry**: a log-normal win is a stronger departure signal than an
alpha win. A shallow alpha-like response can be genuine dysregulation
(under-mobilized) *or* simply a clean, un-elaborated sensory response — so
alpha-wins should be treated as ambiguous, log-normal-wins as more pointed.

### E. Validation direction

Testable hypothesis on any labelled dataset: **gamma should dominate in
well-regulated responses; non-gamma (especially log-normal) should be enriched
under dysregulation.** If `best_model` / `departure_from_gamma` separates groups,
the species/departure descriptor earns promotion from a sidecar to a primary
output. Until validated prospectively, treat it as a flag worth investigating,
not a verdict — a genuinely parallel-stream paradigm can produce a non-gamma
shape in a well-regulated brain.

---

## Concrete next steps for this repo (the sandbox)

- [ ] **`residual-diagnostics` example** — quantify leftover arc-structure
      (autocorrelation / low-frequency power) in the post-gamma residual, to show
      events-vs-misfit on a worked case.
- [ ] **`departure_from_gamma` / `delta_aic`** on `ArcFit`, surfaced in
      `to_dict()`.
- [ ] **Naming/framing pass** — "carrier" vs. "model" in the API and docs; one
      canonical TCI; species as a distinct descriptor.
- [ ] **Optional: residual decomposition** into smooth + discrete components.

---

## Out of scope (unchanged)

The reference deliberately omits the product-specific machinery built on top of
these descriptors: SVD-based source attribution, the full arc-scaled label
*validation* (topography/polarity gates), normative-cohort comparison, and any
AI-generated interpretation.
