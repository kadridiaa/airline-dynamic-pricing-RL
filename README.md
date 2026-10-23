# Plan or Learn? Dynamic Pricing of Airline Tickets

An airline sells the 10 seats of a flight over 30 days and sets a price every day.
We compare **planning** (Dynamic Programming with a demand model that may be wrong) with
**learning** (tabular Q-learning and SARSA, without any model) to answer one question:

> From which model error does learning without a model outperform planning with a wrong one,
> for a given budget of episodes?

Report: [`report/report.pdf`](report/report.pdf) (LaTeX source: [`report/report.tex`](report/report.tex)).

**Short answer.** Learning only wins for large errors on the customers' willingness to pay:
an under-estimation of 30 % or more (from 10³–10⁴ simulated flights), or an over-estimation of
40 % or more (10⁴–3·10⁴ flights). For moderate errors (−20 % to +30 %), the imperfect DP wins or ties.

## Main results

| Policy (evaluated on the true model) | Expected revenue | % of optimum |
|---|---|---|
| DP, true model (optimum) | 1 193.5 € | 100.0 |
| Rising price 140 → 170 € | 1 166.1 € | 97.7 |
| Fixed price 170 € | 1 146.7 € | 96.1 |
| Q-learning, 30k episodes | 1 141.5 € | 95.6 ± 0.3 |
| Q-learning, 100k episodes | 1 150.3 € | 96.4 ± 0.5 |
| SARSA, 100k episodes | 1 150.2 € | 96.4 ± 0.2 |

Learning methods: mean over 10 seeds ± 95 % confidence interval (Student).

## The problem in one paragraph

- **State** `(c, t)`: `c` seats left (0–10), `t` days to departure (0–30), so 341 states.
- **Actions**: 6 prices, 50, 80, 110, 140, 170 and 200 €.
- **Reward**: price × seats sold that day. The episode ends at departure or when the plane is full. γ = 1.
- **Demand**: two customer types.
  - Leisure customers arrive mostly early and pay 80 € on average.
  - Business customers arrive in the last 10 days and pay 180 € on average.
  - Daily sales follow a Poisson law, capped by the seats left.
- **Model error**: the DP plans with both willingness-to-pay means multiplied by `(1 + x)`, `x` from −50 % to +50 %. Every policy is then scored on the *true* model.

Details: [`docs/spec.md`](docs/spec.md); design decisions and their arguments: [`docs/decisions.docx`](docs/decisions.docx).

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/kadridiaa/airline-dynamic-pricing-RL.git
cd airline-dynamic-pricing-RL
pip install -r requirements.txt
```

## Run the tests

```bash
pytest
```

This runs 56 tests in about 30 s. Among them, the consistency tests check that:
- the DP value lies in the 95 % confidence interval of its simulated revenue;
- no wrong-model DP or baseline beats the true-model DP.

## Reproduce the results

Run every command from the root of the repository. Each script writes its CSV in `results/`
and its figure in `results/figures/`. All results are already committed, so you only need to
rerun a script to check it.

| Experiment | Command | Output | Runtime |
|---|---|---|---|
| Optimal DP policy (Fig. 1) | `python experiments/plot_dp_policy.py` | `policy_dp.png` | seconds |
| Sanity check: 1 000 flights at 110 € | `python experiments/fixed_price_check.py` | console | seconds |
| Q-learning hyper-parameter tuning | `python experiments/tune_qlearning.py` | `tune_qlearning.csv` | minutes |
| **E1** convergence, Q-learning vs SARSA (7 budgets × 10 seeds) | `python experiments/e1_convergence.py --rerun` | `e1_convergence.csv`, `e1_convergence.png` | long (the slowest script) |
| **E2** DP with a wrong model | `python experiments/e2_dp_side.py` | `e2_dp_side.csv`, `e2_dp_side.png` | seconds |
| **E2** tipping-point map (Fig. 2) | `python experiments/e2_crossover.py` | `e2_crossover.csv`, `e2_winner_map.png`, `e2_crossover.png` | seconds (reuses E1) |
| **E3** policy heatmaps DP / Q-learning / SARSA | `python experiments/e3_policy_heatmaps.py` | `e3_policy_heatmaps.png` | minutes |
| **E4** discount factor, DP side | `python experiments/e4_dp_side.py` | `e4_dp_side.csv`, `e4_dp_policies.png` | seconds |
| **E4** discount factor, DP vs Q-learning (10 seeds) | `python experiments/e4_comparison.py` | `e4_comparison.csv`, `e4_comparison.png` | ≈ 10 min |
| 95 % confidence intervals | `python experiments/aggregate.py` | `summary_ci.csv` | seconds |

Notes:
- Without `--rerun`, `e1_convergence.py` only redraws the figure from the existing CSV.
- `e2_crossover.py` reuses the Q-learning and SARSA scores of `e1_convergence.csv`, since learning does not depend on the model error. Run E1 first if that file is missing.

## Repository layout

```
src/
  demand.py      demand model: arrivals, purchase probability, sales distribution, model error
  env.py         simulator (reset / step), the only thing the RL agents see
  dp.py          finite-horizon DP: one backward Bellman sweep
  evaluate.py    exact policy evaluation + Monte-Carlo check
  baselines.py   best fixed price, best linearly rising price
  qlearning.py   tabular Q-learning (epsilon-greedy, decaying step size)
  sarsa.py       tabular SARSA (same settings)
  plots.py       policy heatmaps
experiments/     one script per experiment (see table above)
tests/           pytest unit and consistency tests
results/         CSV results and figures
report/          2-page report (LaTeX + PDF)
docs/            specification, roadmap, decision log
```

## Authors

Imane Amraoui and Dia Eddine Kadri. Reinforcement Learning course (J. Martinet), Polytech SI5-IAID, 2026–2027.

References: Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018 (ch. 4, 6, 8.3);
Gallego & van Ryzin, *Optimal dynamic pricing of inventories with stochastic demand over finite horizons*, Management Science, 1994.
