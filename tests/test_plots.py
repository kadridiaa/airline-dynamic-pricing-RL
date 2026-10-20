"""Tests des figures : la heatmap affiche bien la politique, dans le bon sens."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from plots import plot_policy, policy_table  # noqa: E402


def test_policy_table_from_function():
    table = policy_table(lambda c, t: (c + t) % 6, capacity=3, horizon=4)
    assert table.shape == (4, 5)
    assert table[2, 3] == 5


def test_heatmap_orientation():
    # c = 10 en haut à gauche au premier jour (t = 30) ; c = 1 en bas à droite au dernier jour (t = 1)
    table = np.zeros((11, 31), dtype=int)
    table[10, 30] = 5
    table[1, 1] = 3
    ax = plot_policy(table, colorbar=False)
    shown = ax.images[0].get_array()
    assert shown[0, 0] == 5 and shown[-1, -1] == 3
    plt.close(ax.figure)


def test_unseen_states_are_masked():
    unseen = np.zeros((11, 31), dtype=bool)
    unseen[5, 20] = True
    ax = plot_policy(np.zeros((11, 31), dtype=int), unseen=unseen, colorbar=False)
    shown = ax.images[0].get_array()
    assert np.ma.count_masked(shown) == 1
    plt.close(ax.figure)
