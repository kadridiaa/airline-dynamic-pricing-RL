"""
Heatmap de la politique optimale de la DP (issue #12).

Lancer depuis la racine du repo :  python experiments/plot_dp_policy.py
Sortie : results/figures/policy_dp.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel  # noqa: E402
from dp import solve_dp  # noqa: E402
from plots import plot_policy  # noqa: E402

V, policy = solve_dp(DemandModel())
fig, ax = plt.subplots(figsize=(7.2, 2.9))
plot_policy(policy, ax=ax, title=f"Optimal policy (true-model DP), expected revenue {V[10, 30]:.1f} €")
fig.tight_layout()

out_dir = os.path.join(ROOT, "results", "figures")
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "policy_dp.png")
fig.savefig(out, dpi=200)
print("figure enregistrée :", os.path.normpath(out))
