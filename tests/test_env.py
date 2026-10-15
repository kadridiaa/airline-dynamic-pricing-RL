"""Tests de l'environnement AirlineEnv (docs/spec.md, section 2)."""

import numpy as np
import pytest

from demand import DemandModel, PRICES
from env import AirlineEnv


def run_flight(env, action):
    """Joue un vol entier au même prix ; renvoie la liste des (state, reward, done)."""
    state = env.reset()
    steps = []
    done = False
    while not done:
        next_state, reward, done = env.step(action)
        steps.append((state, next_state, reward, done))
        state = next_state
    return steps


def test_reset_returns_full_plane_at_horizon():
    env = AirlineEnv(DemandModel(), seed=0)
    assert env.reset() == (10, 30)


def test_step_before_reset_raises():
    env = AirlineEnv(DemandModel(), seed=0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_after_end_raises():
    env = AirlineEnv(DemandModel(), seed=0)
    run_flight(env, 2)
    with pytest.raises(RuntimeError):
        env.step(2)


def test_transitions_and_rewards_are_consistent():
    env = AirlineEnv(DemandModel(), seed=0)
    for action in range(len(PRICES)):
        for (c, t), (c2, t2), reward, _ in run_flight(env, action):
            sold = c - c2
            assert t2 == t - 1  # un jour passe à chaque pas
            assert 0 <= sold <= c  # on ne vend pas plus que les sièges restants
            assert reward == PRICES[action] * sold  # récompense = prix × ventes


def test_episode_ends_at_departure_or_full_plane():
    env = AirlineEnv(DemandModel(), seed=0)
    for _ in range(200):
        steps = run_flight(env, 0)
        dones = [done for *_, done in steps]
        assert dones[-1] and not any(dones[:-1])  # done seulement au dernier pas
        c, t = steps[-1][1]
        assert t == 0 or c == 0


def test_same_seed_same_trajectory():
    traj1 = run_flight(AirlineEnv(DemandModel(), seed=7), 3)
    traj2 = run_flight(AirlineEnv(DemandModel(), seed=7), 3)
    assert traj1 == traj2


def test_mean_revenue_matches_expected_value():
    # revenu moyen simulé à prix fixe ≈ revenu espéré calculé avec sales_distribution
    model = DemandModel()
    price_idx, p = 2, PRICES[2]
    V = np.zeros((11, 31))  # V[c, t] = revenu espéré restant à prix fixe
    for t in range(1, 31):
        for c in range(1, 11):
            probs = model.sales_distribution(t, p, c)
            V[c, t] = sum(pr * (p * k + V[c - k, t - 1]) for k, pr in enumerate(probs))

    env = AirlineEnv(model, seed=0)
    revenues = [sum(r for *_, r, _ in run_flight(env, price_idx)) for _ in range(5_000)]
    stderr = np.std(revenues) / np.sqrt(len(revenues))
    assert abs(np.mean(revenues) - V[10, 30]) < 4 * stderr
