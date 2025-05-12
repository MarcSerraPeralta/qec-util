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


def observables_to_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Converts the logical observables of a circuit to detectors."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(f"'circuit' is not a stim.Circuit, but a {type(circuit)}.")

    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue

        targets = instr.targets_copy()
        args = instr.gate_args_copy()
        new_instr = stim.CircuitInstruction("DETECTOR", gate_args=args, targets=targets)
        new_circuit.append(new_instr)

    return new_circuit


def move_observables_to_end(circuit: stim.Circuit) -> stim.Circuit:
    """
    Move all the observable definition to the end of the circuit
    while keeping their relative order.
    """
    new_circuit = stim.Circuit()
    obs = []
    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit (where I am going to define the deterministic observables)
    measurements = []
    for i, instr in enumerate(circuit.flattened()):
        if instr.name == "OBSERVABLE_INCLUDE":
            obs.append(instr)
            measurements.append(circuit[i:].num_measurements)
            continue

        new_circuit.append(instr)

    for k, ob in enumerate(obs):
        new_targets = [t.value - measurements[k] for t in ob.targets_copy()]
        new_targets = [stim.target_rec(t) for t in new_targets]
        new_ob = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", new_targets, ob.gate_args_copy()
        )
        new_circuit.append(new_ob)

    return new_circuit
