"""Numerical checks against independent equations, not implementation mirrors."""
import numpy as np
import pytest
from scipy.special import expit
from scipy.stats import norm

from mhi.models import (gaussian_bayes, gaussian_prediction, window_units,
                        mixture_from_units, delta_units, bernoulli_predictions,
                        fit_gaussian, goodness_of_fit, _fit_window_exact,
                        gaussian_complexity, GAUSSIAN_MODELS)


def slow_runlength_reference(x, h, R):
    """Translate the probability-domain MATLAB Bayes recursion literally."""
    weights = np.array([1.])
    estimates = []
    for t in range(len(x)):
        lengths = np.arange(1, t+2)
        mu_after = np.array([sum(x[t-r+1:t+1])/(R*R+r) for r in lengths])
        if t == 0:
            weights = np.array([1.])
        else:
            previous_lengths = np.arange(1, t+1)
            previous_mu = np.array([sum(x[t-r:t])/(R*R+r) for r in previous_lengths])
            predictive_variances = R*R*(1+1/(R*R+previous_lengths))
            weights = np.r_[h*norm.pdf(x[t], 0, np.sqrt(1+R*R)),
                            (1-h)*norm.pdf(x[t], previous_mu, np.sqrt(predictive_variances))*weights]
            weights /= weights.sum()
        estimates.append(weights@mu_after)
    return np.array(estimates)


@pytest.mark.parametrize("R", [.01, .8, 1., 4.])
def test_exact_bayes_matches_matlab_probability_recursion(R):
    x = np.array([.2, .25, -.1, .5, -.7, -.6, -.8])
    np.testing.assert_allclose(gaussian_bayes(x, .1, R), slow_runlength_reference(x, .1, R), atol=1e-12)


def test_model_nesting_and_window_hand_calculation():
    x = np.array([1., 2., 4., -3., 1.])
    np.testing.assert_allclose(gaussian_prediction("delta_rule", x, .1, 1., [1]), x)
    np.testing.assert_allclose(gaussian_prediction("delta_rule", x, .1, 1., [0]), 0)
    np.testing.assert_allclose(gaussian_prediction("mixture_delta_rules", x, .1, 1., [0, 0]), 0, atol=0)
    a = .23
    np.testing.assert_allclose(gaussian_prediction("mixture_delta_rules", x, .1, 1., [a, a]),
                               gaussian_prediction("delta_rule", x, .1, 1., [a]))
    np.testing.assert_allclose(gaussian_prediction("mixture_sliding_windows", x, .1, 1., [2.5, 2.5]),
                               gaussian_prediction("sliding_window", x, .1, 1., [2.5]))
    # At t=3, r=2.5: (4 + 2 + 0.5*1)/(1+2.5).
    assert window_units(x, np.array([2.5]), 1.)[2, 0] == pytest.approx(6.5/3.5)


def test_mixture_matches_direct_matlab_equation_22():
    x = np.array([.3, .6, -.8, -1., .2, 1.5])
    R, h = .8, .1
    rates = np.array([.7, .12])
    units = delta_units(x, rates)
    gap = 1/rates[1]-1/rates[0]
    q = min(1., 1/gap)
    transition = np.array([[h+(1-h)*(1-q), h], [(1-h)*q, 1-h]])
    p = np.array([1., 0.])
    expected = [units[0, 0]]
    for t in range(1, len(x)):
        p = norm.pdf(x[t], units[t], np.sqrt(R*R*(1+rates)))*(transition@p)
        p /= p.sum()
        expected.append(p@units[t])
    np.testing.assert_allclose(mixture_from_units(x, units, rates, h, R), expected, atol=1e-12)


def test_fractional_window_global_optimizer_against_dense_grid():
    rng = np.random.default_rng(9)
    x = rng.normal(size=30)
    y = window_units(x, np.array([4.37]), .8)[:, 0]
    mask = np.arange(len(x)) >= 5
    fitted = _fit_window_exact(x, y, mask, .8)
    assert fitted == pytest.approx(4.37, abs=1e-9)
    noise_y = y+rng.normal(0, .2, len(x))
    fitted = _fit_window_exact(x, noise_y, mask, .8)
    score = lambda r: np.sum((window_units(x, np.array([r]), .8)[mask, 0]-noise_y[mask])**2)
    assert score(fitted) <= min(map(score, np.linspace(1, 60, 3001)))+1e-10


def test_bernoulli_bayes_matches_independent_two_state_hmm():
    beads = np.array([1, 1, 0, 1, 0, 0, 1])
    h, p = .12, .86
    q = .5
    expected = []
    for b in beads:
        expected.append(q*p+(1-q)*(1-p))
        likelihood_black = p if b else 1-p
        likelihood_white = 1-likelihood_black
        posterior = q*likelihood_black/(q*likelihood_black+(1-q)*likelihood_white)
        q = h+(1-2*h)*posterior
    np.testing.assert_allclose(bernoulli_predictions(beads, h, p)["bayesian"], expected, atol=1e-12)


def test_bernoulli_limits_constant_prior_and_prediction_timing():
    beads = np.array([1, 0, 1, 1, 0])
    for pred in bernoulli_predictions(beads, .5, .86).values():
        np.testing.assert_allclose(pred, .5)
    for pred in bernoulli_predictions(beads, .1, .5).values():
        np.testing.assert_allclose(pred, .5)
    a = bernoulli_predictions(beads, .05, .86)
    b = bernoulli_predictions(1-beads, .05, .86)
    for model in a:
        np.testing.assert_allclose(a[model]+b[model], 1)
        assert a[model][0] == pytest.approx(.5)
    # Signed constant prior after the first black observation.
    assert a["constant_prior"][1] == pytest.approx(.14+.72*.95)


def test_no_model_uses_future_observations():
    x = np.array([.2, .6, -.3, -.1, .8, 2.])
    changed = x.copy()
    changed[4:] = -100
    params = {"memoryless": [.3], "delta_rule": [.3], "sliding_window": [2.5],
              "mixture_delta_rules": [.7, .1], "mixture_sliding_windows": [1.5, 4.5]}
    for model in GAUSSIAN_MODELS:
        np.testing.assert_allclose(gaussian_prediction(model, x, .1, 1., params.get(model, []))[:4],
                                   gaussian_prediction(model, changed, .1, 1., params.get(model, []))[:4])
    for model, pred in bernoulli_predictions(x > 0, .1, .86).items():
        # Prediction at t=4 also cannot use bead 4.
        np.testing.assert_allclose(pred[:5], bernoulli_predictions(changed > 0, .1, .86)[model][:5])


def test_synthetic_learning_rate_recovery():
    rng = np.random.default_rng(22)
    x = rng.normal(size=240)
    alpha = .27
    y = gaussian_prediction("delta_rule", x, .1, .8, [alpha])+rng.normal(0, .015, len(x))
    mask = np.arange(len(x)) >= 40
    fits = fit_gaussian(x, y, mask, .1, .8, seed=43)
    assert fits["delta_rule"]["parameters"][0] == pytest.approx(alpha, abs=.01)
    dr_sse = np.sum((y[mask]-fits["delta_rule"]["prediction"][mask])**2)
    assert np.sum((y[mask]-fits["mixture_delta_rules"]["prediction"][mask])**2) <= dr_sse+1e-10


def test_likelihood_noise_cap_and_parameter_penalty():
    y, pred = np.array([0., 2000.]), np.zeros(2)
    f = goodness_of_fit(y, pred, 3, 500)
    assert f["decision_noise_sd"] == 500
    assert f["decision_noise_at_upper_bound"]
    assert f["bic"] == pytest.approx(-2*f["log_likelihood"]+3*np.log(2))
    assert f["paper_log_evidence_score"] == -f["bic"]
    assert gaussian_complexity("mixture_delta_rules", []) == 129
    assert gaussian_complexity("mixture_sliding_windows", [2.5, 4.5]) == 151
