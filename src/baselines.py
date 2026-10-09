"""
Baselines : stratégies simples, sans planification ni apprentissage (docs/spec.md, section 5).

Elles servent à montrer que la DP et le Q-learning font mieux que « le simple ».
Les meilleurs paramètres sont choisis avec evaluate_exact sur le modèle donné :
avec le vrai modèle, c'est la meilleure stratégie simple possible (une borne haute pour les baselines).
"""

from itertools import combinations_with_replacement

from demand import DemandModel, PRICES
from evaluate import evaluate_exact

N_ACTIONS = len(PRICES)


def fixed_price_policy(action):
    """Toujours le même prix, quel que soit l'état."""
    return lambda c, t: action


def rising_price_policy(start_action, end_action, horizon=30):
    """
    Prix qui monte linéairement avec le temps : start_action au premier jour (t = horizon),
    end_action le dernier jour (t = 1). Ne regarde pas les sièges restants.
    """
    def policy(c, t):
        progress = (horizon - t) / (horizon - 1)  # 0 au premier jour, 1 au dernier
        return round(start_action + progress * (end_action - start_action))
    return policy


def best_fixed_price(model: DemandModel, capacity=10, horizon=30):
    """Essaie les 6 prix ; renvoie (action, Evaluation) du meilleur."""
    results = [(a, evaluate_exact(fixed_price_policy(a), model, capacity, horizon)) for a in range(N_ACTIONS)]
    return max(results, key=lambda r: r[1].revenue)


def best_rising_price(model: DemandModel, capacity=10, horizon=30):
    """Essaie toutes les paires (prix de départ <= prix d'arrivée) ; renvoie ((start, end), Evaluation) de la meilleure."""
    results = [((s, e), evaluate_exact(rising_price_policy(s, e, horizon), model, capacity, horizon))
               for s, e in combinations_with_replacement(range(N_ACTIONS), 2)]
    return max(results, key=lambda r: r[1].revenue)
