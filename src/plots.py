"""
Figures du projet. plot_policy sert pour la heatmap de la DP (#12) et pour E3 (DP vs Q-learning vs SARSA).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from demand import PRICES

# rampe bleue à 6 niveaux : plus c'est foncé, plus c'est cher
PRICE_COLORS = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
SWITCH_COLOR = "#eb6834"
UNSEEN_COLOR = "#e4e3df"


def policy_table(policy, capacity=10, horizon=30):
    """Tableau table[c, t] = action, à partir d'une fonction policy(c, t) ou d'un tableau déjà fait."""
    if callable(policy):
        table = np.zeros((capacity + 1, horizon + 1), dtype=int)
        for c in range(1, capacity + 1):
            for t in range(1, horizon + 1):
                table[c, t] = policy(c, t)
        return table
    return np.asarray(policy)


def plot_policy(policy, ax=None, title=None, capacity=10, horizon=30, switch_day=10,
                unseen=None, colorbar=True):
    """
    Heatmap de la politique : prix choisi selon les sièges restants c (vertical, 10 en haut)
    et les jours restants t (horizontal, t = 30 à gauche : le temps avance vers la droite).

    policy : fonction policy(c, t) ou tableau policy[c, t]
    unseen : tableau booléen [c, t] optionnel ; True = état jamais visité (grisé), utile pour le Q-learning
    """
    table = policy_table(policy, capacity, horizon)
    grid = table[1:, 1:][::-1, ::-1].astype(float)  # lignes c = C..1, colonnes t = T..1
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 2.9))

    cmap = ListedColormap(PRICE_COLORS)
    cmap.set_bad(UNSEEN_COLOR)
    if unseen is not None:
        grid = np.ma.masked_where(np.asarray(unseen)[1:, 1:][::-1, ::-1], grid)
    image = ax.imshow(grid, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, len(PRICES) + 0.5), len(PRICES)),
                      aspect="auto")

    ax.set_xticks(range(0, horizon, 2), [str(t) for t in range(horizon, 0, -2)])
    ax.set_yticks(range(capacity), [str(c) for c in range(capacity, 0, -1)])
    ax.set_xlabel("jours restants t (le temps avance vers la droite)")
    ax.set_ylabel("sièges restants c")
    ax.set_xticks(np.arange(-0.5, horizon), minor=True)
    ax.set_yticks(np.arange(-0.5, capacity), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if switch_day:
        ax.axvline(horizon - switch_day - 0.5, color=SWITCH_COLOR, linewidth=2)  # entre t = 11 et t = 10
    if title:
        ax.set_title(title, loc="left", fontsize=10)
    if colorbar:
        cb = ax.figure.colorbar(image, ax=ax, ticks=range(len(PRICES)), pad=0.02)
        cb.ax.set_yticklabels([f"{p} €" for p in PRICES])
        cb.outline.set_visible(False)
    return ax
