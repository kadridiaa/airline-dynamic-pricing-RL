"""
Environnement : la simulation d'UN vol (= un épisode). Voir docs/spec.md, section 2.

Interface volontairement proche de Gym (vu en cours) : reset() puis step(action) en boucle.
"""

import numpy as np

from demand import DemandModel, PRICES


class AirlineEnv:
    def __init__(self, demand_model: DemandModel, capacity=20, horizon=30, seed=None):
        self.demand = demand_model
        self.capacity = capacity
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.seats_left = None
        self.days_left = None

    def reset(self):
        """Début d'un vol : tous les sièges sont libres, t = T. Renvoie l'état (c, t)."""
        self.seats_left = self.capacity
        self.days_left = self.horizon
        return (self.seats_left, self.days_left)

    def step(self, action):
        """
        action : indice du prix dans PRICES (0 à 5).

        Déroulé d'une journée (spec section 2.1) :
          1. tirer les ventes avec self.demand.sample_sales(...)
          2. récompense = prix * ventes
          3. mettre à jour les sièges restants, puis t <- t - 1
          4. done = (t == 0) ou (sièges == 0)

        Renvoie (next_state, reward, done).
        """
        if self.days_left is None:
            raise RuntimeError("Appeler reset() avant step().")
        if self.days_left == 0 or self.seats_left == 0:
            raise RuntimeError("Le vol est terminé : appeler reset().")

        price = PRICES[action]

        # 1. ventes du jour (le prix et le jour influencent la demande)
        sales = self.demand.sample_sales(self.days_left, price, self.seats_left, self.rng)

        # 2. récompense = revenu du jour
        reward = float(price * sales)

        # 3. mise à jour de l'état
        self.seats_left -= sales
        self.days_left -= 1

        # 4. fin d'épisode : départ de l'avion ou avion plein
        done = self.days_left == 0 or self.seats_left == 0

        return (self.seats_left, self.days_left), reward, done
