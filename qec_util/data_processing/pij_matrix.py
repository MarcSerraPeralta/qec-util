from copy import deepcopy

import xarray as xr
import numpy as np


def get_pij_matrix(
    defects: xr.DataArray | np.ndarray,
    avoid_nans: bool = True,
) -> np.ndarray:
    """
    Returns the Pij matrix.

    For the theory behind this formulat see Eq. 11 from the article
    "Exponential suppression of bit or phase errors with cyclic error correction"
    by Google Quantum AI, found in the Supplementary information, accessible from
    https://doi.org/10.1038/s41586-021-03588-y.

    Parameters
    ----------
    defects
        Defect observations. If it is a np.ndarray, it must have shape
        ``(num_shots, num_defects)``. If it is a xr.DataArray, it must have
        ``qec_round``, ``anc_qubit``, ``shot`` as coordinates.
    avoid_nans
        If ``True``, ensures that the values inside square roots are positive.

    Returns
    -------
    pij
        Pij matrix, with entries ``pij[i,j] = pij[j,i]`` corresponding to the
        :math:`p_{ij}` values. If ``defects`` is a numpy array, the rows and
        columns follow the same defect ordering as ``defects``. If ``defects``
        is a xr.DataArray, the defect ordering follows: [ancilla 1 round 1,
        ancilla 1 round 2, ..., ancilla 2 round 1, ancilla 2 round 2, ...],
        with the ancillas following the same ordering as ``defects.anc_qubit``.
    """
    if isinstance(defects, xr.DataArray):
        # convert to defect vector with ordering: A1 r1, A1 r2, ..., A2 r1, A2 r2, ...
        defects = defects.transpose("shot", "anc_qubit", "qec_round").values
        num_shots = defects.shape[0]
        defects = defects.reshape(num_shots, -1)
    if not isinstance(defects, np.ndarray):
        raise TypeError(
            f"'defects' must be a np.ndarray or xr.DataArray, but {type(defects)} was given."
        )

    num_shots, _ = defects.shape

    # obtain <didj> and <di>
    didj = np.einsum("ni, nj -> ij", defects, defects, dtype=np.int32) / num_shots
    di = np.average(defects, axis=0)
    di_matrix = np.repeat(di[np.newaxis, :], len(di), axis=0)

    # get pij using Eq. 11 from SI of https://doi.org/10.1038/s41586-021-03588-y
    numerator = 4 * (didj - di_matrix * di_matrix.T)
    denominator = 1 - 2 * di_matrix - 2 * di_matrix.T + 4 * didj
    tmp = 1 - numerator / denominator

    if avoid_nans:
        tmp[tmp < 0] = 0

    pij = 0.5 - 0.5 * np.sqrt(tmp)

    return pij


def plot_pij_matrix(
    ax,
    pij: np.ndarray,
    qubit_labels: list[str] | None = None,
    num_rounds: int | None = None,
    max_prob: float | int = 0.05,
):
    """
    Plot the given Pij matrix.

    Replicates the Pij matrix plot in Figure 2c from:
    Google Quantum AI. Exponential suppression of bit or phase errors with cyclic
    error correction. Nature 595, 383–387 (2021). https://doi.org/10.1038/s41586-021-03588-y

    Parameters
    ----------
    ax
        Matplotlib axis to plot the pij matrix.
    pij
        Pij matrix.
    qubit_labels
        Labels of the qubits following the same ordering as ``pij``.
        If ``qubit_labels`` and ``num_rounds`` are specified and the shape of
        ``pij`` matches ``(num_defects, num_defects)`` with
        ``num_defects = len(qubits_labels) * num_rounds``, the x- and y-axis
        labels are set up.
    num_rounds
        Number of rounds in ``pij``.
        If ``qubit_labels`` and ``num_rounds`` are specified and the shape of
        ``pij`` matches ``(num_defects, num_defects)`` with
        ``num_defects = len(qubits_labels) * num_rounds``, the x- and y-axis
        labels are set up.
    max_prob
        Maximum probability for the upper triangular ``pij`` when displaying the
        plot. The lower triangular ``pij`` is not clipped.

    Notes
    -----
    This function requires ``matplotlib``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # rotate matrix 90 regrees anticlock-wise
    upper_mask = np.rot90(np.triu(np.ones_like(pij)).astype(bool))
    lower_mask = np.rot90(np.tril(np.ones_like(pij)).astype(bool))
    pij = np.rot90(deepcopy(pij))

    extent = [0, len(pij), 0, len(pij)]

    # plot the upper triangle with the 'Blues' colormap
    colorbar_full = ax.imshow(
        np.ma.array(pij, mask=lower_mask),  # mask invalidates the given elements
        cmap="Blues",
        interpolation="nearest",  # "none" does not work
        vmin=0,
        extent=extent,
    )

    # plot the lower triangle with the 'Reds' colormap
    colorbar_zoom = ax.imshow(
        np.ma.array(pij, mask=upper_mask),  # mask invalidates the given elements
        cmap="Reds",
        interpolation="nearest",  # "none" does not work
        vmin=0,
        vmax=max_prob,
        extent=extent,
    )

    # add a colorbars to the right of the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.3)
    colorbar = plt.colorbar(colorbar_zoom, cax=cax)
    colorbar.ax.yaxis.set_ticks_position("left")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.8)
    colorbar = plt.colorbar(colorbar_full, cax=cax)
    colorbar.ax.yaxis.set_label_position("right")

    # set labels at the center of their regions
    if (num_rounds is not None) and (qubit_labels is not None):
        if pij.shape != (num_rounds * len(qubit_labels),) * 2:
            raise ValueError(
                "The specified 'pij', 'num_rounds', and 'qubit_labels' are not compatible."
            )

        tick_positions_small = np.arange(0, len(pij) + 1, num_rounds)
        tick_positions_big = np.arange(num_rounds / 2, len(pij) + 1, num_rounds)
        ax.set_xticks(tick_positions_big)
        ax.set_xticks(tick_positions_small, minor=True)
        ax.set_yticks(tick_positions_big)
        ax.set_yticks(tick_positions_small, minor=True)

        # set tick labels (optional)
        ax.set_xticklabels(qubit_labels)
        ax.set_yticklabels(qubit_labels)

        ax.tick_params(axis="both", which="major", size=0)
        ax.tick_params(axis="both", which="minor", size=5)

    return ax
