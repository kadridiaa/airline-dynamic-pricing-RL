"""Tests du Q-learning tabulaire (docs/spec.md, sections 5 et 9)."""

import numpy as np
import pytest

from demand import DemandModel, PRICES
from dp import as_policy, solve_dp
from evaluate import evaluate_exact
from qlearning import (N_ACTIONS, epsilon_greedy, epsilon_schedule, greedy_policy,
                       train_qlearning)


@pytest.fixture(scope="module")
def model():
    return DemandModel()


def test_q_shape(model):
    Q = train_qlearning(model, n_episodes=10, capacity=10, horizon=30)
    assert Q.shape == (11, 31, N_ACTIONS)


def test_epsilon_schedule_bounds():
    assert epsilon_schedule(0, 1000) == pytest.approx(1.0)
    assert epsilon_schedule(999, 1000) == pytest.approx(0.01, rel=0.05)
    eps = [epsilon_schedule(e, 1000) for e in range(1000)]
    assert all(b <= a for a, b in zip(eps, eps[1:]))  # décroissante


def test_epsilon_greedy_valid_action(model):
    rng = np.random.default_rng(0)
    Q = np.zeros((11, 31, N_ACTIONS))
    actions = {epsilon_greedy(Q, 5, 15, 0.5, rng) for _ in range(200)}
    assert actions <= set(range(N_ACTIONS))  # que des prix valides
    # avec epsilon = 0 et Q = 0 partout, np.argmax renvoie toujours 0
    assert epsilon_greedy(Q, 5, 15, 0.0, rng) == 0


def test_deterministic_with_same_seed(model):
    Q1 = train_qlearning(model, n_episodes=50, seed=7, capacity=5, horizon=10)
    Q2 = train_qlearning(model, n_episodes=50, seed=7, capacity=5, horizon=10)
    assert np.array_equal(Q1, Q2)


def test_greedy_policy_compatible_with_evaluate(model):
    Q = train_qlearning(model, n_episodes=50, seed=0, capacity=5, horizon=10)
    policy = greedy_policy(Q)
    ev = evaluate_exact(policy, model, capacity=5, horizon=10)
    assert ev.revenue > 0  # la politique s'évalue sans erreur


def test_qlearning_approaches_optimum(model):
    # Cohérence spec n°3 (version réduite pour la vitesse) : avec assez d'épisodes,
    # le Q-learning retrouve (presque) l'optimum de la DP.
    V, _ = solve_dp(model, capacity=5, horizon=10)
    optimum = V[5, 10]
    Q = train_qlearning(model, n_episodes=8000, alpha=0.2, seed=0,
                        capacity=5, horizon=10)
    revenue = evaluate_exact(greedy_policy(Q), model, capacity=5, horizon=10).revenue
    assert revenue >= 0.90 * optimum
