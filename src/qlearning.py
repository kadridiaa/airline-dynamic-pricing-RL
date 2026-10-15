"""
Q-learning tabulaire (docs/spec.md, sections 5 et 9) — côté « Learn » (Dia Eddine).

Contrairement à la DP (src/dp.py), l'agent NE connaît PAS le modèle de demande :
il apprend uniquement en vendant des billets dans l'environnement (essai-erreur).
C'est un algorithme « model-free » et « off-policy » (Sutton & Barto, ch. 6) :
la mise à jour utilise max_a' Q(s', a'), même si l'action suivie était exploratoire.

Décisions validées (spec, section 9) : Q = 0 partout à l'init ; epsilon-greedy avec
epsilon qui part de 1 puis décroît exponentiellement jusqu'à 0.01 ; alpha DÉCROISSANT
(alpha / (1 + alpha_decay * episode / n_episodes)) — un alpha constant ne converge pas
dans cet environnement stochastique (les Q-valeurs oscillent et l'argmax change sans
arrêt) ; budgets d'épisodes sur une échelle logarithmique (~100 → 100 000).
Les valeurs finales d'alpha et de la décroissance d'epsilon sont fixées dans l'issue #14.
"""

import numpy as np

from demand import DemandModel, PRICES
from env import AirlineEnv

N_ACTIONS = len(PRICES)


def epsilon_greedy(Q, c, t, epsilon, rng):
    """
    Choisit une action : avec proba epsilon, un prix au hasard (exploration) ;
    sinon le prix glouton argmax_a Q[c, t, a] (exploitation). En cas d'égalité,
    np.argmax prend le premier, ce qui est déterministe.
    """
    if rng.random() < epsilon:
        return int(rng.integers(N_ACTIONS))
    return int(np.argmax(Q[c, t]))


def epsilon_schedule(episode, n_episodes, epsilon_start=1.0, epsilon_min=0.01):
    """
    Epsilon décroît exponentiellement de epsilon_start vers epsilon_min sur le
    budget d'épisodes : à la fin de l'entraînement, l'agent est (presque) glouton.
    """
    decay = (epsilon_min / epsilon_start) ** (1.0 / n_episodes)
    return max(epsilon_min, epsilon_start * (decay ** episode))


def train_qlearning(model: DemandModel, n_episodes, alpha=0.5, alpha_decay=40.0,
                    epsilon_start=1.0, epsilon_min=0.01,
                    gamma=1.0, seed=0, capacity=10, horizon=30):
    """
    Entraîne un Q-learning tabulaire et renvoie la table Q de taille
    (capacity + 1, horizon + 1, N_ACTIONS).

    Mise à jour (Sutton & Barto, éq. 6.7) :
        Q(c,t,a) += alpha_t * (r + gamma * max_a' Q(c',t',a') - Q(c,t,a))
    avec cible = r seul si l'épisode est terminé (V = 0 au terminal, comme la DP),
    et alpha_t = alpha / (1 + alpha_decay * episode / n_episodes) : alpha décroît
    de `alpha` jusqu'à alpha / (1 + alpha_decay) en fin d'entraînement.
    (alpha_decay = 0 -> alpha constant, utile pour les ablations.)

    seed : reproductibilité totale (chaque seed = une courbe d'apprentissage).
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((capacity + 1, horizon + 1, N_ACTIONS))
    for episode in range(n_episodes):
        epsilon = epsilon_schedule(episode, n_episodes, epsilon_start, epsilon_min)
        alpha_t = alpha / (1 + alpha_decay * episode / n_episodes)
        env = AirlineEnv(model, capacity=capacity, horizon=horizon,
                         seed=int(rng.integers(2 ** 31)))
        c, t = env.reset()
        done = False
        while not done:
            a = epsilon_greedy(Q, c, t, epsilon, rng)
            (c2, t2), reward, done = env.step(a)
            # max_a' Q(s', a') = 0 si terminal (Q vaut 0 aux états terminaux, jamais visités)
            target = reward + (0.0 if done else gamma * np.max(Q[c2, t2]))
            Q[c, t, a] += alpha_t * (target - Q[c, t, a])
            c, t = c2, t2
    return Q


def greedy_policy(Q):
    """
    Politique gloutonne déduite de Q : le format attendu par evaluate.py,
    policy(c, t) -> indice d'action. C'est elle qu'on évalue (pas la politique
    d'entraînement epsilon-greedy).
    """
    return lambda c, t: int(np.argmax(Q[c, t]))
