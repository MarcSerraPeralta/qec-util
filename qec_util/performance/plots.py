import numpy as np


def plot_line_threshold(
    ax,
    phys_prob: np.ndarray,
    log_prob: np.ndarray,
    log_prob_lower: np.ndarray | None = None,
    log_prob_upper: np.ndarray | None = None,
    **kargs,
):
    """Plots the logical error probability as a function of the physiscal
    error probability, including its upper and lower error bars (if given).

    Parameters
    ----------
    ax
        Matplotlib axis in which to plot.
    phys_prob
        Physical error probabilities.
    log_prob
        Logical error probability.
    log_prob_lower
        Lower bound on the logical error probability uncertainty.
    log_prob_upper
        Upper bound on the logical error probability uncertainty.
    **kargs
        Arguments for ``ax.plot``.

    Returns
    -------
    ax
        Matplotlib axis.

    Notes
    -----
    This function requires ``matplotlib``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    kargs_ = dict(marker=".", linestyle="none")
    kargs_.update(kargs)

    p = ax.plot(phys_prob, log_prob, **kargs_)
    color = p[0].get_color()
    if (log_prob_lower is not None) and (log_prob_upper is not None):
        ax.fill_between(
            phys_prob, log_prob_lower, log_prob_upper, color=color, alpha=0.1
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error probability, $p$")
    ax.set_ylabel("logical error probability, $p_L$")

    if kargs_.get("label") is not None:
        ax.legend(loc="best")

    return ax
