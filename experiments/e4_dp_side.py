"""
E4, côté « Plan » (issue #19) : un agent « impatient » (gamma < 1) brade-t-il ses sièges ?

Pour chaque gamma, la DP calcule la politique optimale du critère ACTUALISÉ, puis cette politique
est évaluée sur le vrai modèle avec la métrique commune : le revenu total NON actualisé (ce que la
compagnie encaisse vraiment). On mesure aussi quand les sièges sont vendus : avant ou après
l'arrivée des clients affaires (t <= 10). Tout est exact, donc pas besoin de seeds.

Lancer depuis la racine du repo :  python experiments/e4_dp_side.py
Sorties : results/e4_dp_side.csv et results/figures/e4_dp_policies.png
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from demand import DemandModel, PRICES  # noqa: E402
from dp import as_policy, solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402
from plots import plot_policy  # noqa: E402

GAMMAS = [1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0]  # contient les 3 valeurs de la spec (1 ; 0,9 ; 0,5)
SHOWN = [1.0, 0.9, 0.5]
CAPACITY, HORIZON, SWITCH_DAY = 10, 30, 10


def sales_before_switch(policy_table, model):
    """Sièges vendus en moyenne avant l'arrivée des clients affaires (jours t > 10), par propagation exacte."""
    dist = np.zeros(CAPACITY + 1)  # dist[c] = P(c sièges restants) au début du jour courant
    dist[CAPACITY] = 1.0
    for t in range(HORIZON, SWITCH_DAY, -1):
        new = np.zeros(CAPACITY + 1)
        new[0] = dist[0]
        for c in range(1, CAPACITY + 1):
            probs = model.sales_distribution(t, PRICES[policy_table[c, t]], c)
            for k, pr in enumerate(probs):
                new[c - k] += dist[c] * pr
        dist = new
    return CAPACITY - dist @ np.arange(CAPACITY + 1)


model = DemandModel()
optimum = solve_dp(model)[0][CAPACITY, HORIZON]
rows, tables = [], {}
for gamma in GAMMAS:
    _, table = solve_dp(model, gamma=gamma)
    tables[gamma] = table
    ev = evaluate_exact(as_policy(table), model)
    early = sales_before_switch(table, model)
    rows.append({"gamma": gamma, "revenue": ev.revenue, "pct": 100 * ev.revenue / optimum,
                 "load_factor": ev.load_factor, "sold_out": ev.sold_out,
                 "seats_sold_before_t10": early, "seats_sold_from_t10": ev.load_factor * CAPACITY - early})

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
csv_path = os.path.join(ROOT, "results", "e4_dp_side.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow({k: round(v, 4) for k, v in r.items()})

print(f"{'gamma':>6} {'revenu':>9} {'% opt':>7} {'remplis.':>9} {'plein':>7} {'vendus t>10':>12} {'vendus t<=10':>13}")
for r in rows:
    print(f"{r['gamma']:6.2f} {r['revenue']:8.1f}€ {r['pct']:6.1f}% {100 * r['load_factor']:8.1f}% "
          f"{100 * r['sold_out']:6.1f}% {r['seats_sold_before_t10']:12.2f} {r['seats_sold_from_t10']:13.2f}")

fig, axes = plt.subplots(len(SHOWN), 1, figsize=(7.2, 2.3 * len(SHOWN)), sharex=True)
for ax, gamma in zip(axes, SHOWN):
    r = next(r for r in rows if r["gamma"] == gamma)
    plot_policy(tables[gamma], ax=ax, colorbar=False,
                title=f"γ = {gamma:g}: {r['pct']:.1f}% of optimal revenue, "
                      f"{r['seats_sold_before_t10']:.1f} seats sold before t = 10")
    if gamma != SHOWN[-1]:
        ax.set_xlabel("")
fig.tight_layout(rect=(0, 0, 0.9, 1))
cbar_ax = fig.add_axes((0.91, 0.25, 0.018, 0.5))
cb = fig.colorbar(axes[0].images[0], cax=cbar_ax, ticks=range(len(PRICES)))
cb.ax.set_yticklabels([f"{p} €" for p in PRICES])
cb.outline.set_visible(False)
fig_path = os.path.join(ROOT, "results", "figures", "e4_dp_policies.png")
fig.savefig(fig_path, dpi=200)
print("résultats :", os.path.normpath(csv_path), "|", os.path.normpath(fig_path))
