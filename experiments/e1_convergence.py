"""
E1 — Convergence : Q-learning vs SARSA (comble C13/C14).

Pour chaque budget d'épisodes (échelle log) et chaque seed (10),
on entraîne Q-learning ET SARSA avec les mêmes hyperparamètres,
puis on évalue EXACTEMENT la politique gloutonne sur le vrai modèle.

Lancer depuis la racine du repo :  python experiments/e1_convergence.py
(si results/e1_convergence.csv existe, la figure est seulement redessinée ; --rerun pour tout recalculer)
Sorties : results/e1_convergence.csv  et  results/figures/e1_convergence.png
"""

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel  # noqa: E402
from dp import solve_dp  # noqa: E402
from baselines import best_rising_price  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from qlearning import greedy_policy, train_qlearning  # noqa: E402
from sarsa import train_sarsa  # noqa: E402

BUDGETS = [100, 300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = list(range(10))
ALPHA, ALPHA_DECAY = 0.5, 40.0

true_model = DemandModel()
V, _ = solve_dp(true_model)
optimum = V[10, 30]
print(f"Optimum DP-vrai : {optimum:.0f} EUR | budgets {BUDGETS} x {len(SEEDS)} seeds x 2 méthodes")

METHODS = {
    "qlearning": train_qlearning,
    "sarsa": train_sarsa,
}

csv_path = os.path.join(ROOT, "results", "e1_convergence.csv")
RERUN = "--rerun" in sys.argv or not os.path.exists(csv_path)

rows = []
t0 = time.time()
for method_name, train_fn in (METHODS.items() if RERUN else []):
    print(f"\n=== {method_name.upper()} ===")
    for budget in BUDGETS:
        pcts = []
        for seed in SEEDS:
            Q = train_fn(true_model, n_episodes=budget, alpha=ALPHA,
                         alpha_decay=ALPHA_DECAY, seed=seed)
            rev = evaluate_exact(greedy_policy(Q), true_model).revenue
            pct = 100 * rev / optimum
            pcts.append(pct)
            rows.append({"method": method_name, "budget": budget, "seed": seed,
                         "revenue": round(rev, 1), "pct_optimum": round(pct, 2)})
        print(f"  budget={budget:>7} -> {np.mean(pcts):5.1f}% ± {np.std(pcts):4.1f} "
              f"(min {np.min(pcts):5.1f}%)  [{time.time() - t0:.0f}s]")

# --- Sauvegarde CSV (ou relecture si on ne fait que redessiner) ---
os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
if RERUN:
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
else:
    with open(csv_path, encoding="utf-8") as f:
        rows = [{"method": r["method"], "budget": int(r["budget"]), "seed": int(r["seed"]),
                 "revenue": float(r["revenue"]), "pct_optimum": float(r["pct_optimum"])} for r in csv.DictReader(f)]
    print(f"Figure redessinée à partir de {os.path.normpath(csv_path)} (--rerun pour tout recalculer)")

# --- Figure comparative ---
BLUE, ORANGE, GRID = "#2a78d6", "#e67e22", "#e4e3df"
fig, ax = plt.subplots(figsize=(7.2, 3.8))

for method_name, color, marker in [("qlearning", BLUE, "o"), ("sarsa", ORANGE, "s")]:
    method_rows = [r for r in rows if r["method"] == method_name]
    means = [np.mean([r["pct_optimum"] for r in method_rows if r["budget"] == b]) for b in BUDGETS]
    stds = [np.std([r["pct_optimum"] for r in method_rows if r["budget"] == b]) for b in BUDGETS]
    ax.errorbar(BUDGETS, means, yerr=stds, color=color, linewidth=2,
                marker=marker, markersize=6, markeredgecolor="white", markeredgewidth=1,
                capsize=3, label=f"{'Q-learning' if method_name == 'qlearning' else 'SARSA'} (mean ± 1 sd, 10 seeds)")

rising_pct = 100 * best_rising_price(true_model)[1].revenue / optimum
ax.axhline(rising_pct, color="#52514e", linewidth=1.2, linestyle=(0, (1, 2)))
ax.text(BUDGETS[0], rising_pct + 0.3, f"best rising price: {rising_pct:.1f}%", fontsize=8, color="#52514e", va="bottom")
ax.axhline(95, color="grey", linewidth=1, linestyle="--", alpha=0.6)
ax.text(BUDGETS[0], 95.5, "95% of optimum", fontsize=8, color="grey")
ax.axhline(100, color="grey", linewidth=0.8, linestyle=":", alpha=0.4)
ax.set_xscale("log")
ax.set_xlabel("training episodes (log scale)")
ax.set_ylabel("revenue (% of optimum)")
ax.set_ylim(55, 101)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=9)
ax.set_title("E1: learning curves of Q-learning and SARSA", loc="left", fontsize=10)
fig.tight_layout()
fig_path = os.path.join(ROOT, "results", "figures", "e1_convergence.png")
fig.savefig(fig_path, dpi=200)

print(f"\nDurée totale : {time.time() - t0:.0f}s")
print(f"Résultats : {os.path.normpath(csv_path)} | {os.path.normpath(fig_path)}")
