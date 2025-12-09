from itertools import product
import numpy as np
import xarray as xr
from xarray import DataArray


def get_syndromes(anc_meas: DataArray) -> DataArray:
    """Returns the syndrome outcomes from the ancilla outcomes
    for a memory experiment.

    Parameters
    ----------
    anc_meas
        Ancilla outcomes for the memory experiment.
        Required dimensions: ``qec_round`` and ``meas_reset``.

    Returns
    -------
    syndromes
        Syndrome outcomes.
        Dimensions: same dimensions as ``anc_meas``.
    """
    if anc_meas.meas_reset:
        return anc_meas

    shifted_meas = anc_meas.shift(qec_round=1, fill_value=0)
    syndromes = anc_meas ^ shifted_meas
    return syndromes


def get_defects(syndromes: DataArray, frame: DataArray | None = None) -> DataArray:
    """Returns the defects from the syndrome outcomes for a memory experiment.

    Parameters
    ----------
    syndromes
        Syndrome outcomes of the memory experiment.
        Required dimensions: ``qec_round`` and ``anc_qubit``.
    frame
        Initial Pauli frame of the memory experiment.
        Required dimensions: ``anc_qubit``.

    Returns
    -------
    defects
        Defects.
        Dimensions: same dimensions as ``syndromes``.
    """
    shifted_syn = syndromes.shift(qec_round=1, fill_value=0)

    if frame is not None:
        shifted_syn[dict(qec_round=0)] = frame

    defects = syndromes ^ shifted_syn
    return defects


def get_final_defects(syndromes: DataArray, proj_syndrome: DataArray) -> DataArray:
    """Returns the defects for the final round build from the ancilla outcomes
    and the data qubit outcomes for a memory experiment.

    Parameters
    ----------
    syndromes
        Syndrome outcomes from the ancilla qubits in the memory experiment.
        Required dimensions: ``qec_round`` and ``anc_qubit``.
    proj_syndromes
        Syndrome outcomes built from the data qubit outcomes.
        Required dimensions: ``anc_qubit``.

    Returns
    -------
    defects
        Defects for the stabilizers in ``proj_syndromes``.
        Dimensions: same dimensions as ``syndromes`` without ``qec_round``.
    """
    last_round = syndromes.qec_round.values[-1]
    anc_qubits = proj_syndrome.anc_qubit.values

    last_syndromes = syndromes.sel(anc_qubit=anc_qubits, qec_round=last_round)
    defects = last_syndromes ^ proj_syndrome
    return defects


def get_defect_probs(anc_probs: DataArray, ideal_defects: DataArray) -> DataArray:
    """
    Calculates the probability of observing a defect in the bulk given
    the probabilities of the ancilla qubits being in a given state (0 or 1).

    Parameters
    ----------
    anc_probs
        The probabilities of each ancilla qubits being in a given state (0 or 1) over
        each round of the experiment.
        Required dimensions: ``qec_round``, ``anc_qubit``, and ``state``.
    ideal_defects
        Defect values when the circuit is executed without noise.
        Required dimensions: ``qec_round`` and ``anc_qubit``.

    Returns
    -------
    defect_probs
        The probabilities of observing a defect at each round.
        Dimensions: same dimensions as ``anc_probs`` without ``state``.
    """
    round_shift = 1 if anc_probs.meas_reset else 2
    dims_order = anc_probs.dims

    shifted_probs = anc_probs.shift(qec_round=round_shift)
    prob_product = anc_probs.dot(shifted_probs, dims="state")
    defect_probs = 1 - prob_product.fillna(anc_probs.sel(state=0))
    defect_probs = xr.where(ideal_defects, 1 - defect_probs, defect_probs)
    # reshape into the original dimensions' order because ideal_defects does
    # not have e.g. "shot" dimension and messes the order of the dimensions
    dims_order = (dim for dim in dims_order if dim != "state")
    defect_probs = defect_probs.transpose(*dims_order)

    return defect_probs


def get_final_defect_probs(
    anc_probs: DataArray,
    data_probs: DataArray,
    ideal_final_defects: DataArray,
    proj_mat: DataArray,
) -> DataArray:
    """
    Calculates the final defect probabilities.

    Parameters
    ----------
    anc_probs
        The probabilities of each ancilla qubits being in a given state (0 or 1)
        over each round of the experiment.
        Dimensions: ``shot``, ``qec_round``, ``anc_qubit``, and ``state``.
    data_probs
        The probabilities of each data qubits being in a given state (0 or 1)
        at the end of the experiment.
        Dimensions: ``shot``, ``data_qubit`` and ``state``.
    ideal_final_defects
        Final defect values when the circuit is executed without noise
        Dimensions: ``data_qubit``.
    proj_mat
        The projection matrix mapping the data qubits to the qubits that stabilize them (for
        the basis that the experiment is done in).
        Dimensions: ``data_qubit`` and ``anc_qubit``.

    Returns
    -------
    final_defect_probs
        The final defect probabilities.
        Dimensions: ``shot`` and ``anc_qubit``.
    """
    round_shift = 1 if anc_probs.meas_reset else 2
    comp_rounds = anc_probs.qec_round[-round_shift:]
    comp_probs = anc_probs.sel(qec_round=comp_rounds)

    # relabel to detector for concatenation later on.
    # this was the smartest way I cound figure how to do this in xarray
    comp_probs = comp_probs.rename(qec_round="detector")
    _data_probs = data_probs.rename(data_qubit="detector")
    _proj_mat = proj_mat.rename(data_qubit="detector")

    stab_qubits = proj_mat.anc_qubit.values
    shots = data_probs.shot.values

    data = np.zeros((shots.size, stab_qubits.size))
    final_defect_probs = xr.DataArray(
        data,
        dims=["shot", "anc_qubit"],
        coords=dict(shot=shots, anc_qubit=stab_qubits),
    )

    for ind, stab_qubit in enumerate(stab_qubits):
        proj_vec = _proj_mat.sel(anc_qubit=stab_qubit)
        data_det_probs = _data_probs.where(proj_vec, drop=True)

        anc_det_probs = comp_probs.sel(anc_qubit=stab_qubit)

        probs = xr.concat((data_det_probs, anc_det_probs), dim="detector")

        products = product((0, 1), repeat=probs.detector.size)
        odd_products = filter(lambda x: sum(x) % 2, products)
        combinations = xr.DataArray(
            list(odd_products),
            dims=["ind", "detector"],
            coords=dict(detector=probs.detector),
        )

        comb_probs = xr.where(combinations, probs.sel(state=1), probs.sel(state=0))

        stab_defect_probs = comb_probs.prod(dim="detector").sum(dim="ind")
        final_defect_probs[..., ind] = stab_defect_probs

    final_defect_probs = xr.where(
        ideal_final_defects, 1 - final_defect_probs, final_defect_probs
    )
    # reshape into (shots, qec_round, anc_qubit) because ideal_defects does
    # not have "shot" dimension and messes the order of the dimensions
    final_defect_probs = final_defect_probs.transpose("shot", "anc_qubit")

    return final_defect_probs
