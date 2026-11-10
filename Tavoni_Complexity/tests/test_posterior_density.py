"""Exercise the function defined in Python_Analysis.ipynb directly."""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import multivariate_normal, norm

from mhi.models import gaussian_bayes


@pytest.fixture(scope="module")
def density():
    notebook = json.loads((Path(__file__).resolve().parents[1]/"Python_Analysis.ipynb").read_text())
    cell = next(c for c in notebook["cells"] if c.get("id") == "posterior-source-density")
    namespace = {}
    exec(compile("".join(cell["source"]), "posterior-source-density", "exec"), namespace)
    return namespace["posterior_source_density"]


def test_single_observation_conjugate_posterior(density):
    # Prior N(10, 4), observation variance 9, observed value 16.
    mean = (10/4+16/9)/(1/4+1/9)
    sd = np.sqrt(1/(1/4+1/9))
    grid = np.array([[8., 10.], [12., 16.]])
    expected = norm.pdf(grid, mean, sd)
    actual = density([16.], grid, prior_mean=10, prior_sd=2, noise_ratio=1.5)
    assert actual.shape == grid.shape
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert isinstance(density([16.], 12., prior_mean=10, prior_sd=2, noise_ratio=1.5), float)


@pytest.mark.parametrize("h", [0., .1, .8, 1.])
def test_density_integrates_to_one_and_mean_matches_filter(density, h):
    x = np.array([.3, -.2, .6, -.5, -.8])
    f = lambda z: density(x, z, hazard_rate=h, noise_ratio=.8)
    mass = quad(f, -12, 12, epsabs=1e-10)[0]
    mean = quad(lambda z: z*f(z), -12, 12, epsabs=1e-10)[0]
    assert mass == pytest.approx(1., abs=1e-10)
    assert mean == pytest.approx(gaussian_bayes(x, h, .8)[-1], abs=1e-10)


def test_no_change_and_always_change_closed_forms(density):
    x = np.array([1., -.7, .5])
    grid = np.linspace(-3, 3, 71)
    R = .8
    np.testing.assert_allclose(density(x, grid, hazard_rate=0, noise_ratio=R),
                               norm.pdf(grid, x.sum()/(R*R+len(x)),
                                        np.sqrt(R*R/(R*R+len(x)))))
    np.testing.assert_allclose(density(x, grid, hazard_rate=1, noise_ratio=R),
                               norm.pdf(grid, x[-1]/(R*R+1), np.sqrt(R*R/(R*R+1))))


def test_density_matches_enumeration_of_all_change_histories(density):
    # Integrate the source out of every segment independently, then sum all
    # 2**(T-1) change-point histories. This does not use the online recursion.
    x = np.array([.4, -.6, 1.1, .8])
    grid = np.linspace(-2, 2, 51)
    h, R = .23, .8
    weights, component_pdfs = [], []
    for changes in product([False, True], repeat=len(x)-1):
        starts = [0]+[i+1 for i, change in enumerate(changes) if change]
        stops = starts[1:]+[len(x)]
        probability = h**sum(changes)*(1-h)**(len(changes)-sum(changes))
        for start, stop in zip(starts, stops):
            n = stop-start
            probability *= multivariate_normal.pdf(
                x[start:stop], mean=np.zeros(n), cov=R*R*np.eye(n)+np.ones((n, n)))
        final_segment = x[starts[-1]:]
        mean = final_segment.sum()/(R*R+len(final_segment))
        sd = np.sqrt(R*R/(R*R+len(final_segment)))
        weights.append(probability)
        component_pdfs.append(norm.pdf(grid, mean, sd))
    weights = np.array(weights)/np.sum(weights)
    expected = weights@np.array(component_pdfs)
    np.testing.assert_allclose(density(x, grid, hazard_rate=h, noise_ratio=R), expected, atol=1e-14)


def test_position_is_inclusive_and_future_is_ignored(density):
    x = [.1, .4, -.2, 9.]
    grid = np.linspace(-3, 3, 81)
    expected = density(x[:2], grid)
    np.testing.assert_allclose(density(x, grid, position=1), expected)
    np.testing.assert_allclose(density(x, grid, position=-3), expected)
    np.testing.assert_allclose(density(x[:2]+[np.nan], grid, position=1), expected)
    np.testing.assert_allclose(density(x, grid, position=-1), density(x, grid))
    np.testing.assert_allclose(density(x, grid, position=0), density(x[:1], grid))


def test_density_transforms_with_source_units(density):
    x = np.array([.2, -.4, .8, 1.])
    grid = np.linspace(-3, 3, 91)
    standard = density(x, grid, hazard_rate=.38, noise_ratio=.8)
    original = density(2500+150*x, 2500+150*grid, hazard_rate=.38,
                       noise_ratio=.8, prior_mean=2500, prior_sd=150)
    np.testing.assert_allclose(original, standard/150, atol=1e-14)
    np.testing.assert_array_equal(density(x, [-np.inf, np.inf]), [0., 0.])
    assert density(x, np.empty((0, 2))).shape == (0, 2)


@pytest.mark.parametrize("observations,queries,kwargs,error", [
    ([], 0., {}, ValueError),
    ([[1., 2.]], 0., {}, ValueError),
    ([np.nan], 0., {}, ValueError),
    ([1.], np.nan, {}, ValueError),
    ([1.], 0., {"position": 1}, IndexError),
    ([1.], 0., {"position": -2}, IndexError),
    ([1.], 0., {"position": .5}, TypeError),
    ([1.], 0., {"position": True}, TypeError),
    ([1.], 0., {"hazard_rate": 1.1}, ValueError),
    ([1.], 0., {"noise_ratio": 0}, ValueError),
    ([1.], 0., {"prior_sd": -1}, ValueError),
    ([1.], 0., {"prior_mean": np.inf}, ValueError),
])
def test_invalid_inputs(density, observations, queries, kwargs, error):
    with pytest.raises(error):
        density(observations, queries, **kwargs)
