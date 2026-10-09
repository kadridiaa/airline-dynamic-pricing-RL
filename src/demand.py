"""
Modèle de demande des clients (voir docs/spec.md, section 3).

Le même objet DemandModel sert :
  - à l'environnement, avec les VRAIS paramètres ;
  - à la DP, avec des paramètres FAUSSÉS (erreur de modèle, section 4).

C'est ce qui rend l'expérience E2 simple : une erreur de modèle = un autre DemandModel.
"""

import numpy as np

PRICES = np.array([50, 80, 110, 140, 170, 200])  # actions (indices 0 à 5)


class DemandModel:
    def __init__(self, wtp_mean_leisure=80.0, wtp_mean_business=180.0, switch_day=10):
        """
        wtp_mean_leisure / wtp_mean_business : moyenne du prix max accepté de chaque segment.
        switch_day : à partir de t <= switch_day, les clients affaires arrivent en masse.
        """
        self.wtp_mean_leisure = wtp_mean_leisure
        self.wtp_mean_business = wtp_mean_business
        self.switch_day = switch_day

    def arrival_rates(self, t):
        """Renvoie (lambda_leisure, lambda_business) pour le jour t (jours restants)."""
        if t > self.switch_day:
            return 0.8, 0.1  # loin du départ : surtout des loisirs
        return 0.3, 1.0  # derniers jours : les affaires arrivent en masse

    def purchase_prob(self, price, wtp_mean):
        """P(achat | prix) pour une WTP exponentielle de moyenne wtp_mean."""
        # P(WTP >= prix) = exp(-prix / moyenne) pour une loi exponentielle
        return np.exp(-price / wtp_mean)

    def expected_demand(self, t, price):
        """mu(t, p) = lambda_L(t) * q_L(p) + lambda_B(t) * q_B(p)   (spec section 3.2)."""
        # TODO
        raise NotImplementedError

    def sales_distribution(self, t, price, seats_left):
        """
        Renvoie un vecteur probs de taille seats_left + 1 avec probs[k] = P(ventes = k).

        Utilisé par la DP (qui connaît le modèle). Attention au dernier élément :
        ventes = min(demande, seats_left), donc probs[seats_left] = P(demande >= seats_left).
        Vérification : probs.sum() doit valoir 1.
        Indice : scipy.stats.poisson a .pmf() et .sf() (survie).
        """
        # TODO
        raise NotImplementedError

    def sample_sales(self, t, price, seats_left, rng):
        """
        Tire au sort le nombre de ventes d'une journée (utilisé par l'environnement).
        rng : np.random.Generator (pour la reproductibilité avec les seeds).
        """
        # TODO : tirer une demande Poisson(mu), puis plafonner par seats_left
        raise NotImplementedError

    def with_error(self, x):
        """
        Renvoie un NOUVEAU DemandModel faussé : les deux moyennes de WTP sont multipliées par (1 + x).
        Exemple : true_model.with_error(0.3) -> la DP croit que les clients paient 30 % de plus.
        """
        # TODO
        raise NotImplementedError
