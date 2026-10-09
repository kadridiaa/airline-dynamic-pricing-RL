"""Tests du modèle de demande (docs/spec.md, section 3)."""

import math

import numpy as np
import pytest

from demand import DemandModel, PRICES

CAPACITY = 10
HORIZON = 30


@pytest.fixture
def model():
    return DemandModel()


def test_arrival_rates_switch_at_day_10(model):
    assert model.arrival_rates(30) == (0.8, 0.1)
    assert model.arrival_rates(11) == (0.8, 0.1)
    assert model.arrival_rates(10) == (0.3, 1.0)
    assert model.arrival_rates(1) == (0.3, 1.0)


def test_purchase_prob_is_exponential_survival(model):
    assert model.purchase_prob(0, 80) == pytest.approx(1.0)
    assert model.purchase_prob(80, 80) == pytest.approx(math.exp(-1))
    # plus le prix monte, moins on achète
    probs = [model.purchase_prob(p, 80) for p in PRICES]
    assert all(a > b for a, b in zip(probs, probs[1:]))


def test_expected_demand_by_hand(model):
    # t = 1, p = 110 : 0.3 * e^(-110/80) + 1.0 * e^(-110/180)
    expected = 0.3 * math.exp(-110 / 80) + 1.0 * math.exp(-110 / 180)
    assert model.expected_demand(1, 110) == pytest.approx(expected)


def test_sales_distribution_sums_to_one_everywhere(model):
    for t in range(1, HORIZON + 1):
        for p in PRICES:
            for c in range(CAPACITY + 1):
                assert model.sales_distribution(t, p, c).sum() == pytest.approx(1.0)


def test_sales_distribution_last_entry_absorbs_high_demand(model):
    mu = model.expected_demand(1, 50)
    probs = model.sales_distribution(1, 50, 3)
    poisson = [math.exp(-mu) * mu**k / math.factorial(k) for k in range(3)]
    assert probs[:3] == pytest.approx(poisson)
    assert probs[3] == pytest.approx(1 - sum(poisson))  # P(demande >= 3)


def test_sales_distribution_no_seat_left(model):
    assert model.sales_distribution(5, 110, 0) == pytest.approx([1.0])


def test_sample_sales_never_exceeds_seats(model):
    rng = np.random.default_rng(0)
    assert all(model.sample_sales(1, 50, 1, rng) <= 1 for _ in range(10_000))


def test_sample_sales_matches_sales_distribution(model):
    # ce que vit le Q-learning (tirages) = ce que croit la DP (probabilités)
    rng = np.random.default_rng(0)
    n = 100_000
    sales = np.array([model.sample_sales(1, 50, 3, rng) for _ in range(n)])
    freq = np.bincount(sales, minlength=4) / n
    assert freq == pytest.approx(model.sales_distribution(1, 50, 3), abs=0.01)


def test_sample_sales_reproducible_with_seed(model):
    r1, r2 = np.random.default_rng(42), np.random.default_rng(42)
    draws1 = [model.sample_sales(5, 80, 10, r1) for _ in range(50)]
    draws2 = [model.sample_sales(5, 80, 10, r2) for _ in range(50)]
    assert draws1 == draws2


def test_with_error_returns_new_model(model):
    fake = model.with_error(0.3)
    assert fake.wtp_mean_leisure == pytest.approx(104.0)
    assert fake.wtp_mean_business == pytest.approx(234.0)
    assert model.wtp_mean_leisure == 80.0  # le vrai modèle n'est pas modifié


def test_with_error_zero_is_true_model(model):
    same = model.with_error(0)
    for t in (1, 10, 11, 30):
        for p in PRICES:
            assert same.sales_distribution(t, p, CAPACITY) == pytest.approx(
                model.sales_distribution(t, p, CAPACITY))
