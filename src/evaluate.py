"""
Évaluation commune de toutes les politiques (docs/spec.md, sections 5 et 6).

Règle d'équité : DP, Q-learning, SARSA et baselines sont TOUS évalués par ces fonctions,
sur le VRAI modèle de demande, avec la même métrique (revenu total non actualisé).

Une politique est une fonction  policy(c, t) -> indice d'action (0 à 5),
avec c = sièges restants et t = jours restants. Elle doit être déterministe
(pour le Q-learning : la politique gloutonne argmax_a Q[c, t, a]).
"""

from dataclasses import dataclass

import numpy as np

from demand import DemandModel, PRICES
from env import AirlineEnv


@dataclass
class Evaluation:
    revenue: float  # revenu moyen par vol (€)
    load_factor: float  # part moyenne des sièges vendus (0 à 1)
    sold_out: float  # proportion de vols où l'avion est plein
    revenue_std: float = 0.0  # écart-type du revenu (Monte-Carlo seulement)
    n_flights: int = 0  # 0 = évaluation exacte


def evaluate_exact(policy, model: DemandModel, capacity=10, horizon=30):
    """
    Évaluation exacte par récurrence à rebours (« iterative policy evaluation » du cours,
    en horizon fini : une seule passe de t = 1 à t = T suffit).

      V[c, t] = sum_k P(ventes = k) * (prix * k + V[c - k, t - 1])   revenu espéré restant
      S[c, t] = sum_k P(ventes = k) * (k        + S[c - k, t - 1])   sièges vendus espérés
      F[c, t] = sum_k P(ventes = k) * F[c - k, t - 1]                P(avion plein au départ)

    Conditions terminales : V = S = 0 au départ (t = 0) ou avion plein (c = 0) ; F[0, t] = 1.
    """
    V = np.zeros((capacity + 1, horizon + 1))
    S = np.zeros((capacity + 1, horizon + 1))
    F = np.zeros((capacity + 1, horizon + 1))
    F[0, :] = 1.0
    for t in range(1, horizon + 1):
        for c in range(1, capacity + 1):
            price = PRICES[policy(c, t)]
            probs = model.sales_distribution(t, price, c)
            k = np.arange(c + 1)
            V[c, t] = probs @ (price * k + V[c - k, t - 1])
            S[c, t] = probs @ (k + S[c - k, t - 1])
            F[c, t] = probs @ F[c - k, t - 1]
    return Evaluation(revenue=V[capacity, horizon],
                      load_factor=S[capacity, horizon] / capacity,
                      sold_out=F[capacity, horizon])


def evaluate_mc(policy, model: DemandModel, n_flights=10_000, seed=0, capacity=10, horizon=30):
    """Évaluation par simulation (Monte-Carlo) : on joue n_flights vols dans l'environnement."""
    env = AirlineEnv(model, capacity=capacity, horizon=horizon, seed=seed)
    revenues = np.empty(n_flights)
    seats_sold = np.empty(n_flights)
    for i in range(n_flights):
        c, t = env.reset()
        total, done = 0.0, False
        while not done:
            (c, t), reward, done = env.step(policy(c, t))
            total += reward
        revenues[i] = total
        seats_sold[i] = capacity - c
    return Evaluation(revenue=revenues.mean(),
                      load_factor=seats_sold.mean() / capacity,
                      sold_out=np.mean(seats_sold == capacity),
                      revenue_std=revenues.std(),
                      n_flights=n_flights)
