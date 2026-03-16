import matplotlib.pyplot as plt
import numpy as np

from qec_util.performance.plots import plot_line_threshold, plot_suppression


def test_plot_line_threshold(show_figures):
    phys_prob = np.linspace(1, 4, 10)
    log_prob = np.linspace(1, 6, 10)
    log_prob_lower = log_prob - 1
    log_prob_upper = log_prob + 2

    _, ax = plt.subplots()
    ax = plot_line_threshold(
        ax,
        phys_prob,
        log_prob,
        log_prob_lower,
        log_prob_upper,
        color="red",
        label="example",
    )

    if show_figures:
        plt.show()
    plt.close()

    assert isinstance(ax, plt.Axes)


def test_plot_suppression(show_figures):
    distances = np.array([7, 5, 3], dtype=int)
    log_prob = np.array([1e-4, 1e-3, 1e-2])
    log_prob_lower = log_prob / 1.2
    log_prob_upper = log_prob * 1.3

    _, ax = plt.subplots()
    ax = plot_suppression(
        ax,
        distances,
        log_prob,
        log_prob_lower,
        log_prob_upper,
        color="red",
        label="example",
    )

    if show_figures:
        plt.show()
    plt.close()

    assert isinstance(ax, plt.Axes)
