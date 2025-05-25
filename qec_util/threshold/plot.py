from collections.abc import Iterable

import numpy as np

from .util import get_fit_func, rescale_input


def plot_threshold_fit(
    ax,
    fit_func_name: str,
    popt: np.ndarray,
    rescaled_phys_err_min: float | int = -3,
    rescaled_phys_err_max: float | int = +2,
    color: str | None = "black",
    linestyle: str | None = "-",
    label: str | None = "fit",
    legend_loc: str | None = "upper left",
    **kargs,
):
    """
    Plots the fitted function from ``qec_util.threshold.get_threshold`` fit
    as a function of the rescaled physical error probability.
    It is known as the collapse plot.

    Parameters
    ----------
    ax
        Axes in which to plot the fitted function.
    fit_func_name
        Name of the fit function.
    popt
        Optimized parameters from ``qec_util.threshold.get_threshold``.
        See ``qec_util.threshold.save_fit_information`` for more information.
        Use ``qec_util.threshold.load_fit_information`` to load the fit information
        and extract ``popt``.
    rescaled_phys_err_min
        Miniminum rescaled physical error probability to plot for the fit.
        The rescaling factor is ``(phys_err - popt[0]) * d**(1/popt[1])``,
        with ``d`` the code distance. By default ``-3``.
    rescaled_phys_err_max
        Maximum rescaled physical error probability to plot for the fit.
        The rescaling factor is ``(phys_err - popt[0]) * d**(1/popt[1])``,
        with ``d`` the code distance. By default ``2``.
    legend_loc
        Location of the legend when calling ``ax.legend``.
    **kargs
        Extra arguments for ``ax.plot``. The parameters ``"color"``,
        ``"linestyle"``, and ``"label"`` have been specified by default.

    Returns
    -------
    ax
        The given axes with the plotted fit.

    Notes
    -----
    This function requires ``matplotlibt``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    fit_func, num_params = get_fit_func(fit_func_name)
    if not isinstance(popt, Iterable):
        raise TypeError(f"'popt' must be an np.ndarray, but {type(popt)} was given.")
    popt = np.array(popt)  # allow for non np.ndarrays
    if len(popt) != num_params + 2:
        raise TypeError(
            f"'popt' must be of lenght {num_params+2}, but {len(popt)} was given."
        )

    x = np.linspace(rescaled_phys_err_min, rescaled_phys_err_max, 100)
    y = fit_func(x, *popt[2:])

    ax.plot(x, y, color=color, linestyle=linestyle, label=label, **kargs)
    ax.set_xlabel("rescaled physical error probability, $(p - p_{thr})d^{1/\\mu}$")
    ax.set_ylabel("logical error probability, $p_L$")

    ylim = ax.get_ylim()
    ylim = [max([ylim[0], -0.05]), min(ylim[1], 1.05)]
    ax.set_ylim(*ylim)

    if label is not None:
        ax.legend(loc=legend_loc)

    return ax


def plot_threshold_data(
    ax,
    data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    popt: np.ndarray,
    colors: dict[int, str | None] | None = None,
    markers: dict[int, str | None] | None = None,
    labels: dict[int, str | None] | None = None,
    linestyles: dict[int, str | None] | None = None,
    legend_loc: str | None = "upper left",
    **kargs,
):
    """
    Plots the data as a function of the rescaled physical error probability.
    It is known as the collapse plot.

    Parameters
    ----------
    ax
        Axes in which to plot the fitted function.
    data
        Same structure as ``data`` given to ``qec_util.threshold.get_threshold``:
        dictionary with keys corresponding to the circuit distance, and
        values corresponding to the tuple of (1) physical error probabilities,
        (2) number of failures, (3) number of samples.
    popt
        Optimized parameters from ``qec_util.threshold.get_threshold``.
        See ``qec_util.threshold.save_fit_information`` for more information.
        Use ``qec_util.threshold.load_fit_information`` to load the fit information
        and extract ``popt``. This function only uses the first two elements from
        ``popt``, which correspond to the threshold and :math:`\\mu`.
    colors
        Dictionary mapping distances to colors for ``ax.plot``.
    markers
        Dictionary mapping distances to markers for ``ax.plot``.
    labels
        Dictionary mapping distances to labels for ``ax.plot``.
    linestyles
        Dictionary mapping distances to linestyles for ``ax.plot``.
    legend_loc
        Location of the legend when calling ``ax.legend``.
    **kargs
        Extra arguments for ``ax.plot`` when plotting the points associated to
        each distance.

    Returns
    -------
    ax
        The given axes with the plotted fit.

    Notes
    -----
    This function requires ``matplotlibt``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    if not isinstance(data, dict):
        raise TypeError(f"'data' must be a dict, but {type(data)} was given.")
    if any(not isinstance(d, int) for d in data):
        raise TypeError("Each key of 'data' must be an int.")
    if any((not isinstance(v, Iterable)) or len(v) != 3 for v in data.values()):
        raise TypeError("Each value of 'data' must be a tuple of length 3.")
    for p, f, n in data.values():
        p, f, n = np.array(p), np.array(f), np.array(n)
        if not (p.shape == f.shape == n.shape):
            raise ValueError(
                "Each numpy array triplet in the values of 'data' must have same shape."
            )
        if not (len(p.shape) == len(f.shape) == len(n.shape) == 1):
            raise ValueError(
                "Each numpy array triplet in the values of 'data' must be a vector."
            )
    if not isinstance(popt, Iterable):
        raise TypeError(f"'popt' must be an np.ndarray, but {type(popt)} was given.")
    popt = np.array(popt)  # allow for non np.ndarrays
    if len(popt) < 2:
        raise TypeError(
            f"'popt' must contain at least two elements, but it has {len(popt)}."
        )
    p_thr, mu = popt[:2]
    if colors is None:
        colors = {d: None for d in data}
    if markers is None:
        markers = {d: "." for d in data}
    if labels is None:
        labels = {d: f"$d={d}$" for d in data}
    if linestyles is None:
        linestyles = {d: "none" for d in data}
    if not (set(colors) == set(markers) == set(labels) == set(linestyles) == set(data)):
        raise TypeError(
            "'colors', 'markers', 'labels', 'linestyles', and 'data' must have the same keys."
        )

    x_min, x_max = np.inf, -np.inf
    for distance in data:
        phys_err, num_failures, num_samples = data[distance]
        # convert to np.ndarrays to allow to give it as Sequences
        phys_err = np.array(phys_err)
        num_failures, num_samples = np.array(num_failures), np.array(num_samples)

        log_err = num_failures / num_samples
        rescaled_phys_err = rescale_input(phys_err, distance, p_threshold=p_thr, mu=mu)

        ax.plot(
            rescaled_phys_err,
            log_err,
            color=colors[distance],
            marker=markers[distance],
            label=labels[distance],
            linestyle=linestyles[distance],
            **kargs,
        )

        x_min = min([x_min, rescaled_phys_err.min()])
        x_max = max([x_max, rescaled_phys_err.max()])

    ax.set_xlabel("rescaled physical error probability, $(p - p_{thr})d^{1/\\mu}$")
    ax.set_ylabel("logical error probability, $p_L$")

    margin = 0.05 * (x_max - x_min)
    ax.set_xlim(x_min - margin, x_max + margin)

    if set(labels.values()) != set([None]):
        ax.legend(loc=legend_loc)

    return ax
