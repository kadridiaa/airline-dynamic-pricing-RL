"""
E3 — Heatmaps comparatives des politiques : DP vs Q-learning vs SARSA.

Montre visuellement comment chaque méthode fixe les prix en fonction
du temps restant (t) et des sièges restants (c).

Lancer depuis la racine du repo :  python experiments/e3_policy_heatmaps.py
Sortie : results/figures/e3_policy_heatmaps.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel, PRICES  # noqa: E402
from dp import solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from plots import plot_policy  # noqa: E402
from qlearning import greedy_policy, train_qlearning  # noqa: E402
from sarsa import train_sarsa  # noqa: E402

CAPACITY, HORIZON = 10, 30
BUDGET = 100_000
ALPHA, ALPHA_DECAY = 0.5, 40.0

true_model = DemandModel()
V, dp_table = solve_dp(true_model)
optimum = V[CAPACITY, HORIZON]

# Entraîner Q-learning et SARSA avec le meilleur budget
Q_ql = train_qlearning(true_model, n_episodes=BUDGET, alpha=ALPHA, alpha_decay=ALPHA_DECAY, seed=0)
Q_sa = train_sarsa(true_model, n_episodes=BUDGET, alpha=ALPHA, alpha_decay=ALPHA_DECAY, seed=0)

# Construire les tables de politique pour Q-learning et SARSA
ql_table = np.argmax(Q_ql, axis=2)
sa_table = np.argmax(Q_sa, axis=2)
# États sans aucune information apprise (Q = 0 pour toutes les actions) : l'argmax y vaut 0
# (50 €) par défaut, ce qui n'est pas un choix de l'agent ; on les grise dans la figure.
ql_unseen = np.all(Q_ql == 0, axis=2)
sa_unseen = np.all(Q_sa == 0, axis=2)
print(f"États non appris : Q-learning {ql_unseen[1:, 1:].sum()} / 300, SARSA {sa_unseen[1:, 1:].sum()} / 300")

# Évaluer
from dp import as_policy  # noqa: E402
ev_dp = evaluate_exact(as_policy(dp_table), true_model)
ev_ql = evaluate_exact(greedy_policy(Q_ql), true_model)
ev_sa = evaluate_exact(greedy_policy(Q_sa), true_model)

print(f"DP-vrai  : {100*ev_dp.revenue/optimum:.1f}% | load={100*ev_dp.load_factor:.1f}%")
print(f"Q-learn  : {100*ev_ql.revenue/optimum:.1f}% | load={100*ev_ql.load_factor:.1f}%")
print(f"SARSA    : {100*ev_sa.revenue/optimum:.1f}% | load={100*ev_sa.load_factor:.1f}%")

# --- Figure : 3 heatmaps côte à côte ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

titles = [
    f"DP (optimal): {100*ev_dp.revenue/optimum:.1f}%",
    f"Q-learning ({BUDGET//1000}k episodes): {100*ev_ql.revenue/optimum:.1f}%",
    f"SARSA ({BUDGET//1000}k episodes): {100*ev_sa.revenue/optimum:.1f}%",
]

for ax, table, unseen, title in zip(axes, [dp_table, ql_table, sa_table], [None, ql_unseen, sa_unseen], titles):
    plot_policy(table, ax=ax, colorbar=False, title=title, unseen=unseen)
for ax in axes[1:]:
    ax.set_ylabel("")
fig.text(0.5, 0.005, "grey: states never learned (Q = 0)", ha="center", fontsize=8, color="#52514e")

# Colorbar commune
fig.tight_layout(rect=(0, 0, 0.92, 1))
cbar_ax = fig.add_axes((0.93, 0.18, 0.015, 0.65))
cb = fig.colorbar(axes[0].images[0], cax=cbar_ax, ticks=range(len(PRICES)))
cb.ax.set_yticklabels([f"{p} €" for p in PRICES])
cb.outline.set_visible(False)

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
fig_path = os.path.join(ROOT, "results", "figures", "e3_policy_heatmaps.png")
fig.savefig(fig_path, dpi=200)
print(f"Figure : {os.path.normpath(fig_path)}")
