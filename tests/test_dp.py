"""Tests de la DP (docs/spec.md, section 5). Les tests de cohérence 1 et 4 de la spec sont dans #11."""

import math

import numpy as np
import pytest

from demand import DemandModel, PRICES
from dp import as_policy, solve_dp
from evaluate import evaluate_exact


@pytest.fixture(scope="module")
def model():
    return DemandModel()


@pytest.fixture(scope="module")
def solution(model):
    return solve_dp(model)


def test_terminal_conditions(solution):
    V, _ = solution
    assert np.all(V[:, 0] == 0)  # départ : plus rien à gagner
    assert np.all(V[0, :] == 0)  # avion plein : plus rien à vendre


def test_by_hand_one_day_one_seat(model):
    # 1 siège, 1 jour : V* = max_p p * P(demande >= 1)
    V, policy = solve_dp(model, capacity=1, horizon=1)
    best = [p * (1 - math.exp(-model.expected_demand(1, p))) for p in PRICES]
    assert V[1, 1] == pytest.approx(max(best))
    assert policy[1, 1] == int(np.argmax(best))


def test_value_equals_exact_evaluation_of_its_policy(model, solution):
    # la valeur calculée par la DP = ce que rapporte vraiment sa politique (sur le même modèle)
    V, policy = solution
    assert evaluate_exact(as_policy(policy), model).revenue == pytest.approx(V[10, 30])


def test_value_increases_with_seats_and_days(solution):
    V, _ = solution
    assert np.all(np.diff(V[1:, 1:], axis=0) >= 0)  # un siège de plus ne fait jamais perdre
    assert np.all(np.diff(V, axis=1) >= -1e-9)  # un jour de vente de plus non plus


def test_price_does_not_increase_with_more_seats(solution):
    # structure classique en revenue management : plus il reste de sièges, moins on vend cher
    _, policy = solution
    assert np.all(policy[1:-1, 1:] >= policy[2:, 1:])


def test_discount_lowers_value(model, solution):
    V1, _ = solution
    V09, _ = solve_dp(model, gamma=0.9)
    assert V09[10, 30] < V1[10, 30]
