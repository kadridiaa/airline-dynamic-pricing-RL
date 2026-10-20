"""
Dynamic Programming : politique optimale quand on connaît le modèle (docs/spec.md, section 5).

Équation d'optimalité de Bellman (cours RL3) appliquée à notre MDP, avec s = (c, t) :

  V*(c, t) = max_a  sum_k P(ventes = k | c, t, a) * [ prix(a) * k + gamma * V*(c - k, t - 1) ]

En horizon fini, t fait partie de l'état et diminue de 1 à chaque pas : V*(., t) ne dépend que
de V*(., t - 1). Une seule passe « à rebours » de t = 1 à t = T donne donc la valeur exacte,
sans itérer jusqu'à convergence comme la value iteration en horizon infini.

« DP-vrai » = solve_dp(vrai modèle) ; « DP-faux » = solve_dp(modèle.with_error(x)).
"""

import numpy as np

from demand import DemandModel, PRICES


def solve_dp(model: DemandModel, capacity=10, horizon=30, gamma=1.0):
    """
    Renvoie (V, policy), deux tableaux de taille (capacity + 1, horizon + 1) :
      V[c, t]      = revenu espéré optimal restant (selon `model`)
      policy[c, t] = indice du prix optimal dans l'état (c, t)
    Conditions terminales : V[c, 0] = 0 (départ) et V[0, t] = 0 (avion plein).
    """
    V = np.zeros((capacity + 1, horizon + 1))
    policy = np.zeros((capacity + 1, horizon + 1), dtype=int)
    for t in range(1, horizon + 1):
        for c in range(1, capacity + 1):
            k = np.arange(c + 1)
            # Q*(s, a) pour chacun des 6 prix
            q = [model.sales_distribution(t, p, c) @ (p * k + gamma * V[c - k, t - 1]) for p in PRICES]
            policy[c, t] = int(np.argmax(q))
            V[c, t] = q[policy[c, t]]
    return V, policy


def as_policy(table):
    """Transforme un tableau policy[c, t] en fonction policy(c, t), le format de evaluate.py."""
    return lambda c, t: int(table[c, t])
