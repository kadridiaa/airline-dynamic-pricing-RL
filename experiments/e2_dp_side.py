"""
E2, côté « Plan » (issue #17) : que vaut la DP quand son modèle de demande est faux ?

Pour chaque erreur x sur la WTP (décision D6), la DP planifie avec model.with_error(x),
puis sa politique est évaluée EXACTEMENT sur le vrai modèle (règle d'équité).
Pas de hasard ici : la DP et l'évaluation exacte sont déterministes, donc pas besoin de seeds.

Le Q-learning, lui, n'utilise pas le modèle : sa performance ne dépend pas de x, seulement du
nombre d'épisodes. La carte de E2 s'obtiendra en comparant cette courbe au niveau du Q-learning
pour chaque budget d'épisodes.

Lancer depuis la racine du repo :  python experiments/e2_dp_side.py
Sorties : results/e2_dp_side.csv et results/figures/e2_dp_side.png
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
from baselines import best_fixed_price, best_rising_price, fixed_price_policy, rising_price_policy  # noqa: E402
from demand import DemandModel, PRICES  # noqa: E402
from dp import as_policy, solve_dp  # noqa: E402
from evaluate import evaluate_exact  # noqa: E402

ERRORS = np.round(np.arange(-0.50, 0.501, 0.05), 2) + 0.0  # (+ 0.0 évite « -0 ») contient les 7 valeurs de la spec (section 4)

true_model = DemandModel()
V, _ = solve_dp(true_model)
optimum = V[10, 30]
_, oracle_fixed = best_fixed_price(true_model)
_, oracle_rising = best_rising_price(true_model)

rows = []
for x in ERRORS:
    wrong = true_model.with_error(x)
    _, dp_policy = solve_dp(wrong)
    dp = evaluate_exact(as_policy(dp_policy), true_model)
    # baselines réglées avec le MÊME modèle faux que la DP (comparaison équitable), évaluées sur le vrai
    fixed_action, _ = best_fixed_price(wrong)
    (start, end), _ = best_rising_price(wrong)
    fixed = evaluate_exact(fixed_price_policy(fixed_action), true_model)
    rising = evaluate_exact(rising_price_policy(start, end), true_model)
    rows.append({
        "x": x,
        "dp_revenue": dp.revenue, "dp_pct": 100 * dp.revenue / optimum,
        "dp_load_factor": dp.load_factor, "dp_sold_out": dp.sold_out,
        "fixed_price": int(PRICES[fixed_action]), "fixed_pct": 100 * fixed.revenue / optimum,
        "rising_prices": f"{PRICES[start]}-{PRICES[end]}", "rising_pct": 100 * rising.revenue / optimum,
    })

os.makedirs(os.path.join(ROOT, "results", "figures"), exist_ok=True)
csv_path = os.path.join(ROOT, "results", "e2_dp_side.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    for r in rows:
        writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

print(f"optimum (DP-vrai) : {optimum:.1f} €")
print(f"{'x':>6} {'DP-faux':>9} {'remplis.':>9} {'plein':>7} {'fixe (faux)':>14} {'croissant (faux)':>20}")
for r in rows:
    print(f"{r['x']:+6.0%} {r['dp_pct']:8.1f}% {100 * r['dp_load_factor']:8.1f}% {100 * r['dp_sold_out']:6.1f}%"
          f" {r['fixed_price']:5d} € {r['fixed_pct']:5.1f}% {r['rising_prices']:>10} € {r['rising_pct']:5.1f}%")

# --- figure
BLUE, GREY, INK2, GRID = "#2a78d6", "#8a8984", "#52514e", "#e4e3df"
xs = 100 * ERRORS
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.axhline(100 * oracle_rising.revenue / optimum, color=GREY, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(xs[0], 100 * oracle_rising.revenue / optimum - 0.5,
        f"rising-price ref.: {100 * oracle_rising.revenue / optimum:.1f}%",
        ha="left", va="top", fontsize=8, color=INK2)
ax.plot(xs, [r["rising_pct"] for r in rows], color=GREY, linewidth=1.5, label="rising price tuned on the wrong model")
ax.plot(xs, [r["dp_pct"] for r in rows], color=BLUE, linewidth=2, marker="o", markersize=5,
        markeredgecolor="white", markeredgewidth=1, label="DP planned with the wrong model")
ax.set_xlabel("error x on willingness to pay in the model (%)")
ax.set_ylabel("revenue on the true model (% of optimum)")
ax.set_ylim(64, 101)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right", fontsize=8)
ax.set_title("E2, planning side: DP with a wrong demand model", loc="left", fontsize=10)
fig.tight_layout()
fig_path = os.path.join(ROOT, "results", "figures", "e2_dp_side.png")
fig.savefig(fig_path, dpi=200)
print("résultats :", os.path.normpath(csv_path), "|", os.path.normpath(fig_path))
