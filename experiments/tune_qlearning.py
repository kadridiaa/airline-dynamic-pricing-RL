"""
Réglage des hyperparamètres du Q-learning (issue #14, spec section 9).

On cherche (alpha, alpha_decay) sur une petite grille : 3 seeds par configuration,
évaluation EXACTE de la politique gloutonne sur le vrai modèle (règle d'équité).
La meilleure configuration est réinjectée dans la spec (section 9) et utilisée
pour E1 (experiments/e1_episodes_to_95.py) et E2.

Lancer depuis la racine du repo :  python experiments/tune_qlearning.py
Sortie : results/tune_qlearning.csv (+ meilleure config affichée à la fin)
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

N_EPISODES = 20_000
SEEDS = [0, 1, 2]
ALPHAS = [0.2, 0.5, 1.0]
ALPHA_DECAYS = [10.0, 40.0, 160.0]  # alpha_final = alpha / (1 + decay)

true_model = DemandModel()
V, _ = solve_dp(true_model)
optimum = V[10, 30]
print(f"Optimum DP-vrai : {optimum:.0f} EUR | grille {len(ALPHAS)} x {len(ALPHA_DECAYS)} x {len(SEEDS)} seeds")

rows = []
t0 = time.time()
for alpha in ALPHAS:
    for decay in ALPHA_DECAYS:
        pcts = []
        for seed in SEEDS:
            Q = train_qlearning(true_model, n_episodes=N_EPISODES, alpha=alpha,
                                alpha_decay=decay, seed=seed)
            rev = evaluate_exact(greedy_policy(Q), true_model).revenue
            pcts.append(100 * rev / optimum)
        rows.append({"alpha": alpha, "alpha_decay": decay,
                     "mean_pct": float(np.mean(pcts)), "std_pct": float(np.std(pcts)),
                     "min_pct": float(np.min(pcts))})
        print(f"  alpha={alpha:<4} decay={decay:<6} -> {np.mean(pcts):5.1f}% ± {np.std(pcts):4.1f} "
              f"(min {np.min(pcts):5.1f}%)")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "tune_qlearning.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

best = max(rows, key=lambda r: r["min_pct"])  # robuste : on maximise le pire seed
print(f"\nMeilleure config (pire seed max) : alpha={best['alpha']}, "
      f"alpha_decay={best['alpha_decay']} -> {best['mean_pct']:.1f}% ± {best['std_pct']:.1f}%")
print(f"Durée : {time.time() - t0:.0f}s — résultats dans results/tune_qlearning.csv")
print("À reporter dans docs/spec.md section 9 (décision à deux, cf. réunion du 2026-10-09).")
