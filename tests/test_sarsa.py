"""Tests de SARSA tabulaire (docs/spec.md, section 5)."""

import numpy as np
import pytest

from demand import DemandModel
from dp import solve_dp
from evaluate import evaluate_exact
from qlearning import N_ACTIONS, greedy_policy
from sarsa import train_sarsa


@pytest.fixture(scope="module")
def model():
    return DemandModel()


def test_q_shape(model):
    Q = train_sarsa(model, n_episodes=10, capacity=10, horizon=30)
    assert Q.shape == (11, 31, N_ACTIONS)


def test_deterministic_with_same_seed(model):
    Q1 = train_sarsa(model, n_episodes=50, seed=7, capacity=5, horizon=10)
    Q2 = train_sarsa(model, n_episodes=50, seed=7, capacity=5, horizon=10)
    assert np.array_equal(Q1, Q2)


def test_sarsa_approaches_optimum(model):
    # Comme le Q-learning (epsilon -> 0), SARSA doit retrouver (presque) l'optimum.
    V, _ = solve_dp(model, capacity=5, horizon=10)
    optimum = V[5, 10]
    Q = train_sarsa(model, n_episodes=8000, alpha=0.2, seed=0,
                    capacity=5, horizon=10)
    revenue = evaluate_exact(greedy_policy(Q), model, capacity=5, horizon=10).revenue
    assert revenue >= 0.90 * optimum
