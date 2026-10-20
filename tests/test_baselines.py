"""Tests des baselines (docs/spec.md, section 5)."""

import pytest

from baselines import (best_fixed_price, best_rising_price, fixed_price_policy,
                       rising_price_policy)
from demand import DemandModel, PRICES
from evaluate import evaluate_exact


@pytest.fixture(scope="module")
def model():
    return DemandModel()


def test_fixed_price_policy_is_constant():
    policy = fixed_price_policy(3)
    assert all(policy(c, t) == 3 for c in range(1, 11) for t in range(1, 31))


def test_rising_price_goes_from_start_to_end():
    policy = rising_price_policy(1, 4, horizon=30)
    prices = [policy(10, t) for t in range(30, 0, -1)]  # du premier au dernier jour
    assert prices[0] == 1 and prices[-1] == 4
    assert all(a <= b for a, b in zip(prices, prices[1:]))  # ne baisse jamais
    assert policy(1, 15) == policy(10, 15)  # ignore les sièges restants


def test_best_fixed_price_is_the_best_of_the_six(model):
    action, ev = best_fixed_price(model)
    revenues = [evaluate_exact(fixed_price_policy(a), model).revenue for a in range(len(PRICES))]
    assert ev.revenue == pytest.approx(max(revenues))
    assert PRICES[action] == 170


def test_rising_price_at_least_as_good_as_fixed(model):
    # un prix fixe est un cas particulier de prix croissant (départ = arrivée)
    _, fixed = best_fixed_price(model)
    _, rising = best_rising_price(model)
    assert rising.revenue >= fixed.revenue - 1e-9
