"""
E1 (issue #16, spec section 7) : combien d'épisodes faut-il au Q-learning pour
atteindre 95 % de l'optimum ?

Pour chaque budget d'épisodes (échelle logarithmique, spec section 9) et chaque
seed (10 dès le premier lancement, spec section 6), on entraîne un Q-learning
avec les hyperparamètres de l'issue #14 puis on évalue EXACTEMENT la politique
gloutonne sur le vrai modèle.

Lancer depuis la racine du repo :  python experiments/e1_episodes_to_95.py
Sortie : results/e1_episodes.csv (colonnes : budget, seed, revenue, pct_optimum)
"""

import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel  # noqa: E402
from dp import solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from qlearning import greedy_policy, train_qlearning  # noqa: E402

BUDGETS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = list(range(10))
ALPHA, ALPHA_DECAY = 0.5, 40.0  # issus de experiments/tune_qlearning.py (issue #14)

true_model = DemandModel()
V, _ = solve_dp(true_model)
optimum = V[10, 30]
print(f"Optimum DP-vrai : {optimum:.0f} EUR | budgets {BUDGETS} x {len(SEEDS)} seeds")

rows = []
t0 = time.time()
for budget in BUDGETS:
    pcts = []
    for seed in SEEDS:
        Q = train_qlearning(true_model, n_episodes=budget, alpha=ALPHA,
                            alpha_decay=ALPHA_DECAY, seed=seed)
        rev = evaluate_exact(greedy_policy(Q), true_model).revenue
        pct = 100 * rev / optimum
        pcts.append(pct)
        rows.append({"budget": budget, "seed": seed,
                     "revenue": round(rev, 1), "pct_optimum": round(pct, 2)})
    print(f"  budget={budget:>7} -> {np.mean(pcts):5.1f}% ± {np.std(pcts):4.1f} "
          f"(min {np.min(pcts):5.1f}%)  [{time.time() - t0:.0f}s]")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "e1_episodes.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

means = {b: np.mean([r["pct_optimum"] for r in rows if r["budget"] == b]) for b in BUDGETS}
reached = [b for b in BUDGETS if means[b] >= 95.0]
print(f"\n95 % de l'optimum atteint (en moyenne) à partir de {reached[0]} épisodes."
      if reached else "\n95 % non atteint sur cette échelle de budgets.")
print(f"Durée totale : {time.time() - t0:.0f}s — résultats dans results/e1_episodes.csv")
