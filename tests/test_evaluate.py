"""Tests de la fonction d'évaluation commune (docs/spec.md, sections 5 et 6)."""

import math

import numpy as np
import pytest

from demand import DemandModel, PRICES
from evaluate import evaluate_exact, evaluate_mc


@pytest.fixture
def model():
    return DemandModel()


def fixed_price(action):
    return lambda c, t: action


def rising_price(c, t):
    """Politique qui dépend de (c, t) : 80 € loin du départ, 170 € à la fin, 200 € s'il reste peu de sièges."""
    if c <= 2:
        return 5
    return 1 if t > 10 else 4


def test_exact_by_hand_one_day_one_seat(model):
    # 1 siège, 1 jour, 110 € : revenu = 110 * P(demande >= 1) = 110 * (1 - e^-mu)
    mu = model.expected_demand(1, 110)
    ev = evaluate_exact(fixed_price(2), model, capacity=1, horizon=1)
    assert ev.revenue == pytest.approx(110 * (1 - math.exp(-mu)))
    assert ev.sold_out == pytest.approx(1 - math.exp(-mu))
    assert ev.load_factor == pytest.approx(1 - math.exp(-mu))


def test_exact_bounds(model):
    for a in range(len(PRICES)):
        ev = evaluate_exact(fixed_price(a), model)
        assert 0 <= ev.load_factor <= 1
        assert 0 <= ev.sold_out <= ev.load_factor
        assert 0 < ev.revenue <= PRICES[a] * 10


@pytest.mark.parametrize("policy", [fixed_price(2), fixed_price(4), rising_price])
def test_exact_matches_monte_carlo(model, policy):
    # les deux évaluations doivent donner la même chose (à l'erreur de simulation près)
    exact = evaluate_exact(policy, model)
    mc = evaluate_mc(policy, model, n_flights=5_000, seed=1)
    stderr = mc.revenue_std / np.sqrt(mc.n_flights)
    assert abs(mc.revenue - exact.revenue) < 4 * stderr
    assert mc.load_factor == pytest.approx(exact.load_factor, abs=0.02)
    assert mc.sold_out == pytest.approx(exact.sold_out, abs=0.03)


def test_monte_carlo_reproducible(model):
    a = evaluate_mc(rising_price, model, n_flights=200, seed=3)
    b = evaluate_mc(rising_price, model, n_flights=200, seed=3)
    assert a == b
