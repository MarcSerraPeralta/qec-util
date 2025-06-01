import numpy as np
import matplotlib.pyplot as plt

from qec_util.performance.plots import plot_line_threshold


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
