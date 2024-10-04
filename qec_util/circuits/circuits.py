from collections.abc import Sequence

import stim


def remove_gauge_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Removes the gauge detectors from the given circuit."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"'circuit' is not a stim.Circuit, but a {type(circuit)}.")

    dem = circuit.detector_error_model(allow_gauge_detectors=True)
    gauge_dets = []
    for dem_instr in dem.flattened():
        if dem_instr.type == "error" and dem_instr.args_copy() == [0.5]:
            if len(dem_instr.targets_copy()) != 1:
                raise ValueError("There exist 'composed' gauge detector: {dem_instr}.")
            gauge_dets.append(dem_instr.targets_copy()[0].val)

    if len(gauge_dets) == 0:
        return circuit

    current_det = -1
    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name == "DETECTOR":
            current_det += 1
            if current_det in gauge_dets:
                continue

        new_circuit.append(instr)

    return new_circuit


def remove_detectors_except(
    circuit: stim.Circuit, det_ids_exception: Sequence[int] = []
) -> stim.Circuit:
    """Removes all detectors from a circuit except the specified ones.
    Useful for plotting individual detectors with ``stim.Circuit.diagram``.

    Parameters
    ----------
    circuit
        Stim circuit.
    det_ids_exception
        Index of the detectors to not be removed.

    Returns
    -------
    new_circuit
        Stim circuit without detectors except the ones in ``det_ids_exception``.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"'circuit' is not a stim.Circuit, but a {type(circuit)}.")
    if not isinstance(det_ids_exception, Sequence):
        raise TypeError(
            f"'det_ids_exception' is not a Sequence, but a {type(det_ids_exception)}."
        )
    if any([not isinstance(i, int) for i in det_ids_exception]):
        raise TypeError(
            "'det_ids_exception' is not a sequence of ints, "
            f"{det_ids_exception} was given."
        )

    new_circuit = stim.Circuit()
    current_det_id = -1
    for instr in circuit.flattened():
        if instr.name != "DETECTOR":
            new_circuit.append(instr)
            continue

        current_det_id += 1
        if current_det_id in det_ids_exception:
            new_circuit.append(instr)

    return new_circuit
