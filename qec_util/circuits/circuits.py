from collections.abc import Sequence

import stim

SQ_MEASUREMENTS = ["M", "MX", "MY", "MZ"]
SQ_RESETS = ["R", "RX", "RY", "RZ"]
STIM_INSTRS = (
    ["I", "X", "Y", "Z"]
    + [
        "C_NXYZ",
        "C_NZYX",
        "C_XNYZ",
        "C_XYNZ",
        "C_XYZ",
        "C_ZNYX",
        "C_ZYNX",
        "C_ZYX",
        "H",
        "H_NXY",
        "H_NXZ",
        "H_NYZ",
        "H_XY",
        "H_XZ",
        "H_YZ",
        "S",
        "SQRT_X",
        "SQRT_X_DAG",
        "SQRT_Y",
        "SQRT_Y_DAG",
        "SQRT_Z",
        "SQRT_Z_DAG",
        "S_DAG",
    ]
    + [
        "CNOT",
        "CX",
        "CXSWAP",
        "CY",
        "CZ",
        "CZSWAP",
        "II",
        "ISWAP",
        "ISWAP_DAG",
        "SQRT_XX",
        "SQRT_XX_DAG",
        "SQRT_YY",
        "SQRT_YY_DAG",
        "SQRT_ZZ",
        "SQRT_ZZ_DAG",
        "SWAP",
        "SWAPCX",
        "SWAPCZ",
        "XCX",
        "XCY",
        "XCZ",
        "YCX",
        "YCY",
        "YCZ",
        "ZCX",
        "ZCY",
        "ZCZ",
    ]
    + [
        "CORRELATED_ERROR",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "E",
        "ELSE_CORRELATED_ERROR",
        "HERALDED_ERASE",
        "HERALDED_PAULI_CHANNEL_1",
        "II_ERROR",
        "I_ERROR",
        "PAULI_CHANNEL_1",
        "PAULI_CHANNEL_2",
        "X_ERROR",
        "Y_ERROR",
        "Z_ERROR",
    ]
    + ["M", "MR", "MRX", "MRY", "MRZ", "MX", "MY", "MZ", "R", "RX", "RY", "RZ"]
    + ["MXX", "MYY", "MZZ"]
    + ["MPP", "SPP", "SPP_DAG"]
    + ["REPEAT"]
    + ["DETECTOR", "MPAD", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS", "TICK"]
)


def move_first_resets_to_beginning(circuit: stim.Circuit) -> stim.Circuit:
    """Moves (backwards in time) the first resets for each qubit to appear
    as the first (layer of) operations in the circuit.
    This is a workaround for issue 971 in Stim."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()

    resets: dict[int, None | str] = {i: None for i in range(circuit.num_qubits)}
    for instr in circuit[::-1]:
        name = instr.name if instr.name in SQ_RESETS else None
        for q in instr.targets_copy():
            resets[q.value] = name

    if any(r is None for r in resets.values()):
        raise ValueError(
            "All qubits must have explicit resets before their first operation."
        )

    new_circuit = stim.Circuit()

    # add reset operations at beginning
    split_resets: dict[str, list[int]] = {r: [] for r in SQ_RESETS}
    for q, r in resets.items():
        split_resets[r].append(q)
    for r, qubits in split_resets.items():
        if qubits == []:
            continue
        new_instr = stim.CircuitInstruction(
            name=r, targets=[stim.GateTarget(q) for q in qubits]
        )
        new_circuit.append(new_instr)

    # add remaining operations
    missing_qubits = set(range(circuit.num_qubits))
    for instr in circuit:
        if instr.name not in SQ_RESETS:
            new_circuit.append(instr)
            continue
        if not missing_qubits:
            new_circuit.append(instr)
            continue

        targets = set([t.value for t in instr.targets_copy()])
        new_targets = targets - missing_qubits
        if new_targets:
            new_instr = stim.CircuitInstruction(
                name=instr.name, targets=[stim.GateTarget(t) for t in new_targets]
            )
            new_circuit.append(new_instr)
        missing_qubits.difference_update(targets)

    return new_circuit


def remove_gauge_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Removes the gauge detectors from the given circuit."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )

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


def remove_detectors(
    circuit: stim.Circuit, det_ids_exception: Sequence[int] = []
) -> stim.Circuit:
    """Removes all detectors from a circuit except the specified ones.

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
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if not isinstance(det_ids_exception, Sequence):
        raise TypeError(
            f"'det_ids_exception' is not a Sequence, but a {type(det_ids_exception)}."
        )
    if any(not isinstance(i, int) for i in det_ids_exception):
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


def remove_observables(
    circuit: stim.Circuit, obs_ids_exception: Sequence[int] = []
) -> stim.Circuit:
    """Removes all observables from a circuit except the specified ones.

    Parameters
    ----------
    circuit
        Stim circuit.
    obs_ids_exception
        Index of the observables to not be removed.

    Returns
    -------
    new_circuit
        Stim circuit without observables except the ones in ``obs_ids_exception``.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if not isinstance(obs_ids_exception, Sequence):
        raise TypeError(
            f"'obs_ids_exception' is not a Sequence, but a {type(obs_ids_exception)}."
        )
    if any(not isinstance(i, int) for i in obs_ids_exception):
        raise TypeError(
            "'obs_ids_exception' is not a sequence of ints, "
            f"{obs_ids_exception} was given."
        )

    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue

        if instr.gate_args_copy()[0] in obs_ids_exception:
            new_circuit.append(instr)

    return new_circuit


def observables_to_detectors(
    circuit: stim.Circuit, observables: Sequence[int] | None = None
) -> stim.Circuit:
    """Converts the (specified) logical observables of a circuit to detectors.
    By default, converts all logical observables to detectors.
    It does not move the definition of the observables nor the detectors."""
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    if observables is None:
        observables = list(range(circuit.num_observables))
    if not isinstance(observables, Sequence):
        raise TypeError(f"'observables' is not a sequence, but a {type(observables)}.")
    if any(not isinstance(o, int) for o in observables):
        raise TypeError("All elements in 'observables' must be ints.")
    if min(observables) < 0 or max(observables) > circuit.num_observables:
        raise ValueError("Elements in 'observables' must be valid observable indices.")

    new_circuit = stim.Circuit()
    moved_observables = set()
    for instr in circuit.flattened():
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue
        if instr.gate_args_copy()[0] not in observables:
            new_circuit.append(instr)
            continue

        targets = instr.targets_copy()
        if any(t.is_x_target or t.is_y_target or t.is_z_target for t in targets):
            raise ValueError(
                f"Targets in observable definition cannot be Paulis, but '{instr}' was found."
            )
        obs = instr.gate_args_copy()[0]
        if obs in moved_observables:
            raise ValueError(
                f"Observables cannot be defined in multiple lines, but L{obs} is."
            )
        moved_observables.add(obs)
        new_instr = stim.CircuitInstruction(
            "DETECTOR", gate_args=[obs], targets=targets
        )
        new_circuit.append(new_instr)

    return new_circuit


def redefine_observables(
    circuit: stim.Circuit, new_observables: dict[int, Sequence[int]]
):
    """
    Redefines the observables at the end of the circuit as XORs of
    the currently defined observables.

    Parameters
    ----------
    circuit
        Stim circuit. The observables must be defined at the end of the circuit,
        see ``qec_util.circuits.move_observables_to_end``.
    new_observables
        The indices of the new observables and their corresponding definition
        in terms of list of current observables to XOR.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()
    if not isinstance(new_observables, dict):
        raise TypeError(
            f"'new_observables' must be a dict, but {type(new_observables)} was given."
        )
    if any(not isinstance(o, int) for o in new_observables):
        raise TypeError("All keys in 'new_observables' must be ints.")

    obs_targets, k = {}, 0
    for k, _ in enumerate(circuit):
        if circuit[-k - 1].name != "OBSERVABLE_INCLUDE":
            break
        obs_id = circuit[-k - 1].gate_args_copy()[0]
        # must accumulate targets in a list because observable definitions
        # can be split accross different circuit instructions.
        if obs_id not in obs_targets:
            obs_targets[obs_id] = []
        obs_targets[obs_id] += circuit[-k - 1].targets_copy()

    obs = set([o for os in new_observables.values() for o in os])
    if set(obs_targets) < obs:
        raise ValueError(
            "Observables in 'new_observables' are not present at the end of 'circuit'."
        )

    new_obs_circuit = stim.Circuit()
    for obs_id, old_observables in new_observables.items():
        new_targets = [t for old_obs in old_observables for t in obs_targets[old_obs]]
        new_instr = stim.CircuitInstruction(
            "OBSERVABLE_INCLUDE", gate_args=[obs_id], targets=new_targets
        )
        new_obs_circuit.append(new_instr)

    return circuit[:-k] + new_obs_circuit


def move_observables_to_end(circuit: stim.Circuit) -> stim.Circuit:
    """
    Move all the observable definition to the end of the circuit
    while keeping their relative order.
    """
    if not isinstance(circuit, stim.Circuit):
        raise TypeError(
            f"'circuit' must be a stim.Circuit, but {type(circuit)} was given."
        )
    circuit = circuit.flattened()

    new_circuit = stim.Circuit()
    obs = []
    # moving the definition of the observables messes with the rec[-i] definition
    # therefore I need to take care of how many measurements are between the definition
    # and the end of the circuit (where I am going to define the deterministic observables)
    measurements = []
    for i, instr in enumerate(circuit):
        if instr.name != "OBSERVABLE_INCLUDE":
            new_circuit.append(instr)
            continue

        # observables can be defined with Paulis.
        # if so, their definition must not be moved as it would change the observable.
        if any(
            t.is_x_target or t.is_y_target or t.is_z_target
            for t in instr.targets_copy()
        ):
            # check if observable is already at the end of the circuit
            end = True
            for next_instr in circuit[i:]:
                if next_instr.name != "OBSERVABLE_INCLUDE":
                    end = False
                    break
            if end:
                new_circuit.append(instr)
                continue
            else:
                raise ValueError(
                    f"Observable definition in terms of Paulis found: {instr}."
                )

        obs.append(instr)
        measurements.append(circuit[i:].num_measurements)

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


def remove_non_native_instrs(circuit_str: str) -> str:
    """Removes non-native (flattened) circuit instructions in a string corresponding
    to a Stim circuit."""
    if not isinstance(circuit_str, str):
        raise TypeError(
            f"'circuit_str' must be a str, but {type(circuit_str)} was given."
        )

    new_circuit_str = ""
    for line in circuit_str.split("\n"):
        line = line.lstrip()  # remove initial spaces and tabs
        index1, index2 = line.find(" "), line.find("(")
        index = index1 if index2 == -1 else min(index1, index2)

        if index == -1:
            instruction = line
        else:
            instruction = line[:index]

        if instruction == "REPEAT":
            raise ValueError("'REPEAT' blocks are not supported.")

        if instruction in STIM_INSTRS:
            new_circuit_str += line + "\n"

    # remove extra "\n" added
    new_circuit_str = new_circuit_str[:-1]
    return new_circuit_str
