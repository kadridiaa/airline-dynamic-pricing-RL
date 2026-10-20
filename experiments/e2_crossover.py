"""
E2 — Carte du point de bascule : erreur de modèle × budget → qui gagne ?

C'est LA réponse à la question de recherche du projet :
"À partir de quelle erreur sur le modèle de demande devient-il préférable
 d'apprendre sans modèle (Q-learning / SARSA) plutôt que de planifier
 avec un modèle imparfait (DP) ?"

Pour chaque erreur x et chaque budget d'épisodes :
  - DP-faux planifie avec model.with_error(x), évaluée sur le vrai modèle
  - Q-learning et SARSA apprennent sur le vrai modèle (10 seeds), évalués exactement
  - On compare : qui a le meilleur % de l'optimum ?

Lancer depuis la racine du repo :  python experiments/e2_crossover.py
Sorties : results/e2_crossover.csv
          results/figures/e2_crossover.png  (carte principale)
          results/figures/e2_winner_map.png (carte "qui gagne")
"""

import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel  # noqa: E402
from dp import as_policy, solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from qlearning import greedy_policy, train_qlearning  # noqa: E402
from sarsa import train_sarsa  # noqa: E402

# --- Paramètres ---
ERRORS = np.round(np.arange(-0.50, 0.501, 0.10), 2) + 0.0
BUDGETS = [300, 1_000, 3_000, 10_000, 30_000, 100_000]
SEEDS = list(range(10))
ALPHA, ALPHA_DECAY = 0.5, 40.0

true_model = DemandModel()
V, _ = solve_dp(true_model)
optimum = V[10, 30]
print(f"Optimum DP-vrai : {optimum:.0f} EUR")
print(f"Grille : {len(ERRORS)} erreurs x {len(BUDGETS)} budgets x {len(SEEDS)} seeds x 2 méthodes")

# --- 1. Côté DP-faux (pas de seed, déterministe) ---
dp_results = {}
for x in ERRORS:
    wrong = true_model.with_error(x)
    _, dp_policy = solve_dp(wrong)
    dp_rev = evaluate_exact(as_policy(dp_policy), true_model).revenue
    dp_pct = 100 * dp_rev / optimum
    dp_results[x] = dp_pct
    print(f"  DP-faux x={x:+.0%} -> {dp_pct:.1f}%")

# --- 2. Côté RL (Q-learning + SARSA, pour chaque budget et seed) ---
rl_results = {}  # (method, budget) -> mean_pct
rows = []
t0 = time.time()

e1_csv_path = os.path.join(ROOT, "results", "e1_convergence.csv")
if os.path.exists(e1_csv_path):
    print(f"\n[INFO] Chargement des résultats précalculés depuis {e1_csv_path}...")
    e1_data = {}
    with open(e1_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            m = r["method"]
            b = int(r["budget"])
            pct = float(r["pct_optimum"])
            e1_data.setdefault((m, b), []).append(pct)
    for (m, b), pcts in e1_data.items():
        if b in BUDGETS:
            rl_results[(m, b)] = float(np.mean(pcts))
            print(f"  {m:<10} budget={b:>7} -> {rl_results[(m, b)]:.1f}% ± {np.std(pcts):.1f}% (depuis E1)")
else:
    METHODS = {"qlearning": train_qlearning, "sarsa": train_sarsa}
    for method_name, train_fn in METHODS.items():
        print(f"\n=== {method_name.upper()} ===")
        for budget in BUDGETS:
            pcts = []
            for seed in SEEDS:
                Q = train_fn(true_model, n_episodes=budget, alpha=ALPHA,
                             alpha_decay=ALPHA_DECAY, seed=seed)
                rev = evaluate_exact(greedy_policy(Q), true_model).revenue
                pct = 100 * rev / optimum
                pcts.append(pct)
            mean_pct = np.mean(pcts)
            rl_results[(method_name, budget)] = mean_pct
            print(f"  budget={budget:>7} -> {mean_pct:.1f}% ± {np.std(pcts):.1f}%  [{time.time()-t0:.0f}s]")

        for x in ERRORS:
            ql_pct = rl_results.get(("qlearning", budget), 0)
            sa_pct = rl_results.get(("sarsa", budget), 0)
            best_rl = max(ql_pct, sa_pct)
            best_rl_name = "qlearning" if ql_pct >= sa_pct else "sarsa"
            dp_pct = dp_results[x]
            winner = best_rl_name if best_rl > dp_pct else "dp_faux"

# Construire le CSV final avec toutes les combinaisons
csv_rows = []
for x in ERRORS:
    dp_pct = dp_results[x]
    for budget in BUDGETS:
        ql_pct = rl_results.get(("qlearning", budget), 0)
        sa_pct = rl_results.get(("sarsa", budget), 0)
        best_rl = max(ql_pct, sa_pct)
        best_rl_name = "qlearning" if ql_pct >= sa_pct else "sarsa"
        winner = best_rl_name if best_rl > dp_pct else "dp_faux"
        csv_rows.append({
            "error_x": x, "budget": budget,
            "dp_faux_pct": round(dp_pct, 2),
            "qlearning_pct": round(ql_pct, 2),
            "sarsa_pct": round(sa_pct, 2),
            "winner": winner,
        })

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
csv_path = os.path.join(ROOT, "results", "e2_crossover.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
    w.writeheader()
    w.writerows(csv_rows)

# --- Figure 1 : Courbes croisées ---
BLUE, RED, ORANGE, GREY, GRID = "#2a78d6", "#c0392b", "#e67e22", "#8a8984", "#e4e3df"
fig, ax = plt.subplots(figsize=(8.5, 4.5))

# DP-faux en fonction de l'erreur
errors_pct = 100 * ERRORS
dp_curve = [dp_results[x] for x in ERRORS]
ax.plot(errors_pct, dp_curve, color=RED, linewidth=2.5, marker="D", markersize=5,
        markeredgecolor="white", markeredgewidth=1, label="DP avec modèle faux", zorder=3)

# Lignes horizontales Q-learning et SARSA pour quelques budgets clés
styles = [(1_000, ":", 1.2), (3_000, "--", 1.5), (10_000, "--", 1.8), (30_000, "-", 2.0), (100_000, "-", 2.2)]
for budget, ls, lw in styles:
    ql_pct = rl_results.get(("qlearning", budget), 0)
    sa_pct = rl_results.get(("sarsa", budget), 0)
    ax.axhline(ql_pct, color=BLUE, linewidth=lw, linestyle=ls, alpha=0.7)
    ax.text(errors_pct[-1] + 1, ql_pct, f"QL {budget//1000}k", fontsize=7, color=BLUE, va="center")
    ax.axhline(sa_pct, color=ORANGE, linewidth=lw, linestyle=ls, alpha=0.7)
    ax.text(errors_pct[-1] + 8, sa_pct, f"SA {budget//1000}k", fontsize=7, color=ORANGE, va="center")

ax.set_xlabel("Erreur x sur la WTP dans le modèle (%)")
ax.set_ylabel("Revenu (% de l'optimum)")
ax.set_ylim(60, 101)
ax.set_xlim(errors_pct[0] - 2, errors_pct[-1] + 15)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)

# Légende
from matplotlib.lines import Line2D  # noqa: E402
legend_elements = [
    Line2D([0], [0], color=RED, linewidth=2.5, marker="D", markersize=5, markeredgecolor="white", label="DP-faux"),
    Line2D([0], [0], color=BLUE, linewidth=1.5, label="Q-learning (horizontal = indép. de x)"),
    Line2D([0], [0], color=ORANGE, linewidth=1.5, label="SARSA (horizontal = indép. de x)"),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=8, loc="lower left")
ax.set_title("E2 : Point de bascule — DP-faux vs model-free", loc="left", fontsize=10)
fig.tight_layout()
fig1_path = os.path.join(ROOT, "results", "figures", "e2_crossover.png")
fig.savefig(fig1_path, dpi=200)

# --- Figure 2 : Carte "qui gagne" (heatmap) ---
fig2, ax2 = plt.subplots(figsize=(7.5, 4))
winner_matrix = np.zeros((len(BUDGETS), len(ERRORS)))
for i, budget in enumerate(BUDGETS):
    for j, x in enumerate(ERRORS):
        dp_pct = dp_results[x]
        ql_pct = rl_results.get(("qlearning", budget), 0)
        sa_pct = rl_results.get(("sarsa", budget), 0)
        best_rl = max(ql_pct, sa_pct)
        # 1 = RL gagne, 0 = DP-faux gagne, 0.5 = quasi-égal (< 0.5% diff)
        diff = best_rl - dp_pct
        if diff > 0.5:
            winner_matrix[i, j] = 1.0
        elif diff < -0.5:
            winner_matrix[i, j] = 0.0
        else:
            winner_matrix[i, j] = 0.5

cmap = mcolors.ListedColormap(["#c0392b", "#f0e68c", "#2a78d6"])
bounds = [-0.1, 0.25, 0.75, 1.1]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
im = ax2.imshow(winner_matrix, aspect="auto", cmap=cmap, norm=norm, origin="lower")

ax2.set_xticks(range(len(ERRORS)))
ax2.set_xticklabels([f"{x:+.0%}" for x in ERRORS], fontsize=8)
ax2.set_yticks(range(len(BUDGETS)))
ax2.set_yticklabels([f"{b//1000}k" if b >= 1000 else str(b) for b in BUDGETS], fontsize=9)
ax2.set_xlabel("Erreur x sur la WTP")
ax2.set_ylabel("Budget d'épisodes (RL)")
ax2.set_title("E2 : Qui gagne ? (rouge = DP-faux, bleu = model-free, jaune = ≈ égal)", loc="left", fontsize=9)

# Annoter chaque case
for i, budget in enumerate(BUDGETS):
    for j, x in enumerate(ERRORS):
        dp_pct = dp_results[x]
        ql_pct = rl_results.get(("qlearning", budget), 0)
        sa_pct = rl_results.get(("sarsa", budget), 0)
        best_rl = max(ql_pct, sa_pct)
        diff = best_rl - dp_pct
        ax2.text(j, i, f"{diff:+.1f}", ha="center", va="center", fontsize=7,
                 color="white" if abs(diff) > 2 else "black")

fig2.tight_layout()
fig2_path = os.path.join(ROOT, "results", "figures", "e2_winner_map.png")
fig2.savefig(fig2_path, dpi=200)

print(f"\nDurée totale : {time.time() - t0:.0f}s")
print(f"CSV    : {os.path.normpath(csv_path)}")
print(f"Fig 1  : {os.path.normpath(fig1_path)}")
print(f"Fig 2  : {os.path.normpath(fig2_path)}")
