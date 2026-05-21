"""Path A (3-model AIC) and Path B (gamma-only) fit behavior.

The key faithfulness property: AIC may select any of the three models, but the
descriptor arc (Path B) is *always* a gamma.

Note on model identifiability: the gamma function is a *superset* of the alpha
function (alpha == gamma at n=2) and closely approximates a mild log-normal, so
AIC cannot reliably "recover" alpha from alpha-shaped data. Log-normal wins only
when the true shape is heavy-tailed enough that gamma genuinely can't match it.
These tests assert what is actually true, not a tidier fiction.
"""

import numpy as np

from pyarcparam import fit_all_models, fit_tci, iterative_decompose
from pyarcparam.models import alpha_func, compute_aic, compute_r_squared, gamma_func, lognormal_func

from ._fixtures import gamma_signal, make_synthetic_evoked

# --- pure math helpers -----------------------------------------------------


def test_r_squared_perfect_and_zero():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_r_squared(y, y) == 1.0
    # Predicting the mean explains no variance.
    assert abs(compute_r_squared(y, np.full_like(y, y.mean()))) < 1e-12


def test_aic_rewards_lower_residual():
    n = 100
    assert compute_aic(n, ss_res=1.0, k=3) < compute_aic(n, ss_res=10.0, k=3)
    # A perfect fit (ss_res == 0) returns inf — faithful to CW; real noisy data
    # never hits this, but noiseless synthetic data can.
    assert compute_aic(n, ss_res=0.0, k=3) == float("inf")


# --- Path A: model selection by AIC ---------------------------------------


def _grid():
    return np.linspace(0.004, 0.8, 200)


def test_fit_all_models_recovers_gamma():
    t = _grid()
    y = gamma_func(t, 8.0, 4.0, 0.08)
    _, best, fit = fit_all_models(t, y, peak_amp=y.max(), peak_time=t[np.argmax(y)])
    assert best == "gamma"
    assert fit["r_squared"] > 0.99


def test_gamma_is_a_superset_of_alpha():
    """Alpha data is fit near-perfectly by gamma (alpha == gamma at n=2)."""
    t = _grid()
    y = alpha_func(t, 5.0, 0.15)
    fits, _, _ = fit_all_models(t, y, peak_amp=y.max(), peak_time=t[np.argmax(y)])
    assert fits["gamma"]["r_squared"] > 0.999
    assert fits["alpha"]["r_squared"] > 0.999


def test_heavy_tailed_lognormal_is_selected():
    """A heavy-tailed log-normal shape (sigma=0.7) gamma can't match -> log_normal wins."""
    t = _grid()
    rng = np.random.default_rng(0)
    y = lognormal_func(t, 2.0, np.log(0.25), 0.7) + rng.standard_normal(t.size) * 0.01
    _, best, _ = fit_all_models(t, y, peak_amp=y.max(), peak_time=t[np.argmax(y)])
    assert best == "log_normal"


def test_fit_tci_known_gamma_recovers_high_tci():
    evoked = make_synthetic_evoked(
        signal_func=gamma_signal(A=8.0, n=4.0, tau=0.08),
        noise_level=0.1,
        rng=np.random.default_rng(99),
    )
    result = fit_tci(evoked)
    assert not result["fit_failed"]
    assert result["tci"] is not None and result["tci"] > 0.90
    assert result["best_model"] == "gamma"
    assert result["peak_latency_ms"] > 0


def test_fit_tci_flat_signal_fails():
    evoked = make_synthetic_evoked(noise_level=0.0)  # zero signal
    result = fit_tci(evoked)
    assert result["fit_failed"] is True
    assert result["tci"] is None


# --- Path A vs Path B on the SAME log-normal data -------------------------


def test_lognormal_data_pathA_selects_lognormal_pathB_stays_gamma():
    """On heavy-tailed log-normal data, AIC picks log_normal, yet the descriptor
    arc is still a gamma."""
    t = _grid()
    rng = np.random.default_rng(0)
    y = lognormal_func(t, 2.0, np.log(0.25), 0.7) + rng.standard_normal(t.size) * 0.01

    # Path A: model selection -> log_normal
    _, best_name, _ = fit_all_models(t, y, peak_amp=y.max(), peak_time=t[np.argmax(y)])
    assert best_name == "log_normal"

    # Path B: iterative decompose fits a gamma regardless of true shape
    decomp = iterative_decompose(t, np.maximum(y, 0.0))
    assert decomp is not None
    gamma_fit = decomp["gamma"]
    # A gamma fit always exposes gamma parameters (n, tau) — proof the
    # descriptor arc is gamma even when the data is log-normal.
    assert "n" in gamma_fit and "tau" in gamma_fit
    assert gamma_fit["n"] >= 1.0
    assert gamma_fit["y_pred"] is not None
