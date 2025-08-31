from collections.abc import Sequence

import stim


SQ_MEASUREMENTS = ["M", "MX", "MY", "MZ"]


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


def format_rec_targets(
    circuit: stim.Circuit, qubit_inds: None | dict[str, int] = None
) -> str:
    """
    Returns the string of a circuit where the ``rec[-i]``s in the detectors and observables
    have been replaced/formatted to ``qubit_label[-t]`` with  ``t`` corresponding to the relative
    number of measurements for the specific qubits (not for all qubits stim does with ``i``).

    Parameters
    ----------
    circuit
        Stim circuit.
    qubit_inds
        Mapping of the qubit labels to their corresponding qubit index in ``circuit``.

    Returns
    -------
    circuit_str
        Formatted stim circuit string.

    Notes
    -----
    It only supports circuits with single-qubit-measurement instructions, that is without
    parity-measurement instructions such as ``MZZ`` or ``MPP``.

    See ``format_to_rec_targets`` for the inverse functionality.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if qubit_inds is None:
        qubit_inds = {f"q{i}": i for i in range(circuit.num_qubits)}
    if not isinstance(qubit_inds, dict):
        raise TypeError(
            f"'qubit_inds' must be a dict, but {type(qubit_inds)} was given."
        )
    if any(not isinstance(ind, int) for ind in qubit_inds.values()):
        raise TypeError("The values of 'qubit_inds' must be integers.")
    if any(not isinstance(q, str) for q in qubit_inds.keys()):
        raise TypeError("The keys of 'qubit_inds' must be strings.")
    ind_to_label = {v: k for k, v in qubit_inds.items()}

    circuit_str = ""
    measurements: list[tuple[str, int]] = []
    num_qubit_meas = {q: 0 for q in qubit_inds}
    for instr in circuit.flattened():
        if instr.name in SQ_MEASUREMENTS:
            qubit_labels = [ind_to_label[i.value] for i in instr.targets_copy()]
            for qubit_label in qubit_labels:
                measurements.append((qubit_label, num_qubit_meas[qubit_label]))
                num_qubit_meas[qubit_label] += 1
        if instr.name not in ["DETECTOR", "OBSERVABLE_INCLUDE"]:
            circuit_str += str(instr) + "\n"
            continue

        targets = [i.value for i in instr.targets_copy()]

        targets_str: list[str] = []
        for target in targets:
            meas = measurements[target]
            qubit_label, abs_ind = meas
            rel_ind = abs_ind - num_qubit_meas[qubit_label]
            targets_str.append(f"{qubit_label}[{rel_ind}]")

        # get prefix
        instr_str = str(instr)
        prefix = instr_str.split("rec[")[0]

        circuit_str += prefix + " ".join(targets_str) + "\n"

    return circuit_str


def format_to_rec_targets(circuit_str: str, qubit_inds: dict[str, int]) -> stim.Circuit:
    """
    Returns the stim circuit from the string where the detectors and observables are
    specified with ``qubit_label[-t]``. This corresponds to the inverse of ``format_rec_targets``.

    Parameters
    ----------
    circuit_str
        String corresponding to a Stim circuit, except for the detectors and observables.
    qubit_inds
        Mapping of the qubit labels to their corresponding qubit index in ``circuit_str``.

    Returns
    -------
    circuit
        Corresponding Stim circuit.

    Notes
    -----
    It only supports circuits with single-qubit-measurement instructions, that is without
    parity-measurement instructions such as ``MZZ`` or ``MPP``.
    """
    if not isinstance(circuit_str, str):
        raise TypeError(
            f"'circuit_str' must be a str, but {type(circuit_str)} was given."
        )
    if not isinstance(qubit_inds, dict):
        raise TypeError(
            f"'qubit_inds' must be a dict, but {type(qubit_inds)} was given."
        )
    if any(not isinstance(ind, int) for ind in qubit_inds.values()):
        raise TypeError("The values of 'qubit_inds' must be integers.")
    if any(not isinstance(q, str) for q in qubit_inds.keys()):
        raise TypeError("The keys of 'qubit_inds' must be strings.")
    if any(" " in q for q in qubit_inds.keys()):
        raise TypeError("The keys of 'qubit_inds' cannot contain spaces.")
    ind_to_label = {v: k for k, v in qubit_inds.items()}

    # remove spaces at the beginning and at the end of each instruction
    instructions = [
        line.strip() for line in circuit_str.split("\n") if line.strip() != ""
    ]

    new_circuit_str = ""
    num_meas = 0
    meas_order: dict[str, list[int]] = {q: [] for q in qubit_inds}
    for instr in instructions:
        if not (("DETECTOR" in instr) or ("OBSERVABLE_INCLUDE" in instr)):
            new_circuit_str += instr + "\n"

            stim_instr = stim.Circuit(instr)[0]
            if stim_instr.name in SQ_MEASUREMENTS:
                for target in stim_instr.targets_copy():
                    label = ind_to_label[target.value]
                    meas_order[label].append(num_meas)
                    num_meas += 1

            continue

        # detectors can have or not have the parenthesis after 'DETECTOR'.
        if ")" in instr:
            index = instr.index(")") + 2
        else:
            index = instr.index(" ") + 1

        prefix = instr[:index]
        targets = instr[index:].split(" ")

        new_targets: list[str] = []
        for target in targets:
            label, s2 = target.split("[")
            rel_meas_id = int(s2[:-1])  # because of the trailing ']'.
            abs_meas_id = meas_order[label][rel_meas_id]
            new_targets.append(f"rec[{abs_meas_id - num_meas}]")

        new_circuit_str += prefix + " ".join(new_targets) + "\n"

    return stim.Circuit(new_circuit_str)
