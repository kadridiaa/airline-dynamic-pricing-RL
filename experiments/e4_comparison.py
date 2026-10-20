"""
E4 — Effet de gamma (1.0, 0.9, 0.5) : l'agent devient-il myope ?
Compare la DP et le Q-learning sous différentes valeurs de gamma.

Chaque politique est apprise/calculée avec gamma (critère actualisé),
puis évaluée exactement sur le vrai environnement avec la métrique réelle :
le revenu non actualisé (ce que la compagnie encaisse vraiment).

Lancer depuis la racine : python experiments/e4_comparison.py
Sorties : results/e4_comparison.csv
          results/figures/e4_comparison.png
"""

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel, PRICES  # noqa: E402
from dp import as_policy, solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from qlearning import greedy_policy, train_qlearning  # noqa: E402

GAMMAS = [1.0, 0.9, 0.5]
SEEDS = list(range(5))
BUDGET = 30_000
CAPACITY, HORIZON, SWITCH_DAY = 10, 30, 10

true_model = DemandModel()
V_opt, _ = solve_dp(true_model, gamma=1.0)
optimum = V_opt[CAPACITY, HORIZON]

print(f"Optimum DP-vrai (gamma=1.0) : {optimum:.0f} EUR")
print(f"Comparaison DP vs Q-learning pour gamma in {GAMMAS}...")

def sales_before_switch(policy_fn, model):
    """Calcule exactement le nombre moyen de sièges vendus avant t <= 10."""
    dist = np.zeros(CAPACITY + 1)
    dist[CAPACITY] = 1.0
    for t in range(HORIZON, SWITCH_DAY, -1):
        new = np.zeros(CAPACITY + 1)
        new[0] = dist[0]
        for c in range(1, CAPACITY + 1):
            a = policy_fn(c, t)
            probs = model.sales_distribution(t, PRICES[a], c)
            for k, pr in enumerate(probs):
                new[c - k] += dist[c] * pr
        dist = new
    return CAPACITY - dist @ np.arange(CAPACITY + 1)


rows = []
t0 = time.time()

# 1. DP
for g in GAMMAS:
    _, dp_table = solve_dp(true_model, gamma=g)
    pol = as_policy(dp_table)
    ev = evaluate_exact(pol, true_model)
    early = sales_before_switch(pol, true_model)
    late = ev.load_factor * CAPACITY - early
    pct = 100 * ev.revenue / optimum
    rows.append({
        "method": "DP", "gamma": g, "seed": "exact",
        "revenue": round(ev.revenue, 1), "pct_optimum": round(pct, 2),
        "load_factor": round(ev.load_factor, 3),
        "sold_before_t10": round(early, 2), "sold_after_t10": round(late, 2),
    })
    print(f"  DP  gamma={g:3.1f} -> {pct:5.1f}% | load={100*ev.load_factor:4.1f}% | "
          f"avant t10={early:4.2f} / après={late:4.2f}")

# 2. Q-learning
for g in GAMMAS:
    revs, loads, earlies = [], [], []
    for s in SEEDS:
        Q = train_qlearning(true_model, n_episodes=BUDGET, gamma=g, seed=s)
        pol = greedy_policy(Q)
        ev = evaluate_exact(pol, true_model)
        early = sales_before_switch(pol, true_model)
        late = ev.load_factor * CAPACITY - early
        pct = 100 * ev.revenue / optimum
        revs.append(pct)
        loads.append(ev.load_factor)
        earlies.append(early)
        rows.append({
            "method": "Q-learning", "gamma": g, "seed": s,
            "revenue": round(ev.revenue, 1), "pct_optimum": round(pct, 2),
            "load_factor": round(ev.load_factor, 3),
            "sold_before_t10": round(early, 2), "sold_after_t10": round(late, 2),
        })
    print(f"  QL  gamma={g:3.1f} -> {np.mean(revs):5.1f}% ± {np.std(revs):3.1f}% | "
          f"load={100*np.mean(loads):4.1f}% | avant t10={np.mean(earlies):4.2f} [{time.time()-t0:.0f}s]")

# Sauvegarder CSV
os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
csv_path = os.path.join(ROOT, "results", "e4_comparison.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

# --- Graphique E4 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

gammas_str = [str(g) for g in GAMMAS]
x = np.arange(len(GAMMAS))
width = 0.35

dp_pcts = [r["pct_optimum"] for r in rows if r["method"] == "DP"]
ql_pcts_mean = [np.mean([r["pct_optimum"] for r in rows if r["method"] == "Q-learning" and r["gamma"] == g]) for g in GAMMAS]
ql_pcts_std = [np.std([r["pct_optimum"] for r in rows if r["method"] == "Q-learning" and r["gamma"] == g]) for g in GAMMAS]

# Panel 1: % de l'optimum
ax1.bar(x - width/2, dp_pcts, width, label="DP", color="#c0392b", alpha=0.85)
ax1.bar(x + width/2, ql_pcts_mean, width, yerr=ql_pcts_std, capsize=4, label=f"Q-learning ({BUDGET//1000}k)", color="#2a78d6", alpha=0.85)
ax1.set_ylabel("Revenu (% de l'optimum gamma=1.0)")
ax1.set_xlabel("Facteur d'actualisation gamma")
ax1.set_xticks(x)
ax1.set_xticklabels([f"gamma = {g}" for g in GAMMAS])
ax1.set_ylim(70, 105)
ax1.legend(frameon=False)
ax1.grid(axis="y", color="#e4e3df", linestyle="--")
ax1.set_title("Impact de la myopie sur le revenu", fontsize=10, loc="left")

# Panel 2: Sièges vendus avant t <= 10 (loisirs) vs après t <= 10 (affaires)
dp_early = [r["sold_before_t10"] for r in rows if r["method"] == "DP"]
ql_early = [np.mean([r["sold_before_t10"] for r in rows if r["method"] == "Q-learning" and r["gamma"] == g]) for g in GAMMAS]

ax2.bar(x - width/2, dp_early, width, label="DP : sièges vendus tôt (t > 10)", color="#e67e22", alpha=0.85)
ax2.bar(x + width/2, ql_early, width, label="QL : sièges vendus tôt (t > 10)", color="#27ae60", alpha=0.85)
ax2.axhline(10, color="grey", linestyle=":", label="Capacité max (10 sièges)")
ax2.set_ylabel("Nombre moyen de sièges vendus tôt")
ax2.set_xlabel("Facteur d'actualisation gamma")
ax2.set_xticks(x)
ax2.set_xticklabels([f"gamma = {g}" for g in GAMMAS])
ax2.set_ylim(0, 11)
ax2.legend(frameon=False, fontsize=8)
ax2.grid(axis="y", color="#e4e3df", linestyle="--")
ax2.set_title("Ventes prématurées aux loisirs (myopie)", fontsize=10, loc="left")

for ax in (ax1, ax2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout()
fig_path = os.path.join(ROOT, "results", "figures", "e4_comparison.png")
fig.savefig(fig_path, dpi=200)

print(f"\nDurée totale : {time.time() - t0:.0f}s")
print(f"CSV : {csv_path}")
print(f"Fig : {fig_path}")
