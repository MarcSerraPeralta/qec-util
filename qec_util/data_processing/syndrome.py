from xarray import DataArray


def get_syndromes(anc_meas: DataArray) -> DataArray:
    """Returns the syndrome outcomes from the ancilla outcomes
    for a memory experiment.

    Parameters
    ----------
    anc_meas
        Ancilla outcomes for the memory experiment.
        Expected minimum set of coordinates: ``qec_round`` and ``meas_reset``.

    Returns
    -------
    syndromes
        Syndrome outcomes.
        Coordinates: same coordinates as ``anc_meas``.
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
        Expected coordinates: ``qec_round`` and ``anc_qubit``.
    frame
        Initial Pauli frame of the memory experiment.
        Expected coordinates: ``anc_qubit``.

    Returns
    -------
    defects
        Defects.
        Coordinates: ``qec_round`` and ``anc_qubit``.
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
        Expected coordinates: ``qec_round`` and ``anc_qubit``.
    proj_syndromes
        Syndrome outcomes built from the data qubit outcomes.
        Expected coordinates: ``anc_qubit``.

    Returns
    -------
    defects
        Defects for the stabilizers in ``proj_syndromes``.
        Coordinates: ``anc_qubit``.
    """
    last_round = syndromes.qec_round.values[-1]
    anc_qubits = proj_syndrome.anc_qubit.values

    last_syndromes = syndromes.sel(anc_qubit=anc_qubits, qec_round=last_round)
    defects = last_syndromes ^ proj_syndrome
    return defects
