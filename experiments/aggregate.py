"""
Agrégation multi-seeds + intervalles de confiance (issue #20).

Relit les résultats bruts des 10 seeds (E1, E4) sans rien relancer, et calcule pour chaque
configuration la moyenne et l'intervalle de confiance à 95 % de la moyenne (loi de Student,
n - 1 degrés de liberté). Vérifie aussi, pour E2, si chaque victoire du RL sur la DP-faux
est significative : la DP-faux est déterministe, donc le RL gagne significativement quand
la borne basse de son IC dépasse le score de la DP-faux.

Lancer depuis la racine du repo :  python experiments/aggregate.py
Sortie : results/summary_ci.csv
"""

import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(__file__), "..")
RESULTS = os.path.join(ROOT, "results")


def mean_ci(values, level=0.95):
    """Moyenne et demi-largeur de l'IC de la moyenne (Student)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    half = stats.t.ppf((1 + level) / 2, n - 1) * values.std(ddof=1) / np.sqrt(n)
    return values.mean(), half


def read(name):
    with open(os.path.join(RESULTS, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


summary = []

# --- E1 : Q-learning et SARSA selon le budget d'épisodes
groups = defaultdict(list)
for r in read("e1_convergence.csv"):
    groups[(r["method"], int(r["budget"]))].append(float(r["pct_optimum"]))
for (method, budget), pcts in sorted(groups.items()):
    m, h = mean_ci(pcts)
    summary.append({"experiment": "E1", "method": method, "setting": f"budget={budget}",
                    "n_seeds": len(pcts), "mean_pct": m, "ci95_half_width": h, "ci95_low": m - h, "ci95_high": m + h})

# --- E4 : Q-learning selon gamma (la DP est exacte, pas d'IC)
groups = defaultdict(list)
for r in read("e4_comparison.csv"):
    if r["method"] == "Q-learning":
        groups[float(r["gamma"])].append(float(r["pct_optimum"]))
for gamma, pcts in sorted(groups.items(), reverse=True):
    m, h = mean_ci(pcts)
    summary.append({"experiment": "E4", "method": "qlearning", "setting": f"gamma={gamma:g}",
                    "n_seeds": len(pcts), "mean_pct": m, "ci95_half_width": h, "ci95_low": m - h, "ci95_high": m + h})

with open(os.path.join(RESULTS, "summary_ci.csv"), "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    for row in summary:
        writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()})

print(f"{'exp':<4} {'méthode':<10} {'réglage':<16} {'n':>3} {'moyenne':>8} {'IC 95 %':>18}")
for s in summary:
    print(f"{s['experiment']:<4} {s['method']:<10} {s['setting']:<16} {s['n_seeds']:>3} {s['mean_pct']:7.2f}% "
          f"[{s['ci95_low']:6.2f} ; {s['ci95_high']:6.2f}]")

# --- E2 : les victoires du RL sont-elles significatives ?
ci = {(s["method"], int(s["setting"].split("=")[1])): s for s in summary if s["experiment"] == "E1"}
print("\nE2 : cases où le meilleur RL bat la DP-faux de plus de 0,5 point")
for r in read("e2_crossover.csv"):
    x, budget, dp = float(r["error_x"]), int(r["budget"]), float(r["dp_faux_pct"])
    best = max(("qlearning", "sarsa"), key=lambda m: ci[(m, budget)]["mean_pct"])
    s = ci[(best, budget)]
    if s["mean_pct"] - dp > 0.5:
        verdict = "significatif" if s["ci95_low"] > dp else "NON significatif"
        print(f"  x = {x:+.0%}, {budget:>6} épisodes : {best} {s['mean_pct']:.2f} "
              f"[{s['ci95_low']:.2f} ; {s['ci95_high']:.2f}] vs DP-faux {dp:.2f} -> {verdict}")

rising = 97.7
best100k = max(ci[("qlearning", 100_000)], ci[("sarsa", 100_000)], key=lambda s: s["ci95_high"])
print(f"\nPlafond du RL (100k) : borne haute de l'IC = {best100k['ci95_high']:.2f} % < prix croissant {rising} % : "
      f"{'oui, significatif' if best100k['ci95_high'] < rising else 'non'}")
