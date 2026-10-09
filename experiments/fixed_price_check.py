"""
Test rapide de la roadmap (issue #7) : 1 000 vols à prix fixe 110 €.
Affiche le revenu moyen et le taux de remplissage moyen.

Lancer depuis la racine du repo :  python experiments/fixed_price_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from demand import DemandModel, PRICES  # noqa: E402
from env import AirlineEnv  # noqa: E402

N_FLIGHTS = 1_000
PRICE_IDX = 2  # 110 €
SEED = 0

env = AirlineEnv(DemandModel(), seed=SEED)
revenues, load_factors, sold_out_days = [], [], []

for _ in range(N_FLIGHTS):
    state = env.reset()
    total, done = 0.0, False
    while not done:
        state, reward, done = env.step(PRICE_IDX)
        total += reward
    seats_left, days_left = state
    revenues.append(total)
    load_factors.append(1 - seats_left / env.capacity)
    if seats_left == 0:
        sold_out_days.append(days_left)

revenues = np.array(revenues)
print(f"{N_FLIGHTS} vols à prix fixe {PRICES[PRICE_IDX]} € (C = {env.capacity}, T = {env.horizon}, seed = {SEED})")
print(f"  revenu moyen      : {revenues.mean():7.1f} € ± {1.96 * revenues.std() / np.sqrt(N_FLIGHTS):.1f} (IC 95 %)")
print(f"  remplissage moyen : {100 * np.mean(load_factors):7.1f} %")
print(f"  avion plein       : {100 * len(sold_out_days) / N_FLIGHTS:7.1f} % des vols"
      + (f" (en moyenne à {np.mean(sold_out_days):.1f} jours du départ)" if sold_out_days else ""))
