"""
SARSA tabulaire (docs/spec.md, section 5) — côté « Learn » (Dia Eddine).

Même idée que le Q-learning (src/qlearning.py) : apprendre sans modèle, par
essai-erreur. Différence : SARSA est « on-policy » (Sutton & Barto, ch. 6) —
la mise à jour utilise l'action a' EFFECTIVEMENT choisie à l'état suivant
(par la politique epsilon-greedy), pas le max sur les actions :

        Q(c,t,a) += alpha * (r + gamma * Q(c',t',a') - Q(c,t,a))

Conséquence : SARSA apprend la valeur de la politique exploratoire (plus
« prudente »), là où Q-learning apprend la valeur de la politique gloutonne.
En pratique, avec epsilon qui décroît vers 0, les deux convergent vers Q*.
"""

import numpy as np

from demand import DemandModel, PRICES
from env import AirlineEnv
from qlearning import N_ACTIONS, epsilon_greedy, epsilon_schedule, greedy_policy

__all__ = ["train_sarsa", "greedy_policy"]


def train_sarsa(model: DemandModel, n_episodes, alpha=0.5, alpha_decay=40.0,
                epsilon_start=1.0, epsilon_min=0.01,
                gamma=1.0, seed=0, capacity=10, horizon=30):
    """
    Entraîne un SARSA tabulaire. Même signature que train_qlearning pour que
    les expériences comparent les deux à hyperparamètres égaux.
    Renvoie la table Q de taille (capacity + 1, horizon + 1, N_ACTIONS).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((capacity + 1, horizon + 1, N_ACTIONS))
    for episode in range(n_episodes):
        epsilon = epsilon_schedule(episode, n_episodes, epsilon_start, epsilon_min)
        alpha_t = alpha / (1 + alpha_decay * episode / n_episodes)
        env = AirlineEnv(model, capacity=capacity, horizon=horizon,
                         seed=int(rng.integers(2 ** 31)))
        c, t = env.reset()
        a = epsilon_greedy(Q, c, t, epsilon, rng)
        done = False
        while not done:
            (c2, t2), reward, done = env.step(a)
            a2 = epsilon_greedy(Q, c2, t2, epsilon, rng) if not done else 0
            # Q(c',t',a') = 0 si terminal (états terminaux jamais visités, Q = 0)
            target = reward + (0.0 if done else gamma * Q[c2, t2, a2])
            Q[c, t, a] += alpha_t * (target - Q[c, t, a])
            c, t, a = c2, t2, a2
    return Q
