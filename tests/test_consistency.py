"""
Tests de cohérence de la spec (docs/spec.md, section 8), à passer avant toute expérience.
Le test 3 (Q-learning proche de l'optimum avec beaucoup d'épisodes) est dans tests/test_qlearning.py (#13).
"""

import numpy as np
import pytest

from baselines import best_fixed_price, best_rising_price
from demand import DemandModel
from dp import as_policy, solve_dp
from evaluate import evaluate_exact, evaluate_mc

ERRORS = [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5]  # spec section 4


@pytest.fixture(scope="module")
def model():
    return DemandModel()


@pytest.fixture(scope="module")
def optimum(model):
    V, policy = solve_dp(model)
    return V[10, 30], policy


def test_1_dp_value_matches_monte_carlo(model, optimum):
    # V_T(C) calculée par la DP ≈ revenu moyen simulé de sa politique
    value, policy = optimum
    mc = evaluate_mc(as_policy(policy), model, n_flights=10_000, seed=0)
    half_width = 1.96 * mc.revenue_std / np.sqrt(mc.n_flights)  # IC à 95 %
    assert abs(mc.revenue - value) < half_width


@pytest.mark.parametrize("x", ERRORS)
def test_2_wrong_model_never_beats_true_dp(model, optimum, x):
    # une DP planifiée avec un modèle faux, évaluée sur le vrai modèle, ne peut pas battre DP-vrai
    value, _ = optimum
    _, wrong_policy = solve_dp(model.with_error(x))
    assert evaluate_exact(as_policy(wrong_policy), model).revenue <= value + 1e-9


def test_2_baselines_never_beat_true_dp(model, optimum):
    value, _ = optimum
    assert best_rising_price(model)[1].revenue <= value + 1e-9


def test_4_best_fixed_price_below_dp(model, optimum):
    value, _ = optimum
    assert best_fixed_price(model)[1].revenue < value
