import pytest
import stim

from qec_util.circuits import (
    format_rec_targets,
    format_to_rec_targets,
    merge_observable_definitions,
    move_first_resets_to_beginning,
    move_observables_to_end,
    observables_to_detectors,
    redefine_observables,
    remove_detectors,
    remove_gauge_detectors,
    remove_non_native_instrs,
    remove_observables,
)


def test_move_first_resets_to_beginning():
    circuit = stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        X_ERROR(0.1) 0
        I 0 1
        TICK
        R 0
        Y_ERROR(0.1) 0 2
        RX 1
        X 0
        TICK
        RY 2
        TICK
        M 0
        R 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) Z0
        """
    )

    new_circuit = move_first_resets_to_beginning(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0
        RX 1
        RY 2
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        TICK
        Y_ERROR(0.1) 0
        X 0
        TICK
        TICK
        M 0
        R 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) Z0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        R 0
        X 0
        RX 0 1
        """
    )

    new_circuit = move_first_resets_to_beginning(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0
        RX 1
        X 0
        RX 0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        R 2 3 4 5 6 7
        DEPOLARIZE2(0.1) 2 7 3 6 5 4 0 1
        R 0 1
        """
    )

    new_circuit = move_first_resets_to_beginning(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3 4 5 6 7
        DEPOLARIZE2(0.1) 2 7 3 6 5 4
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit("X 0")

    with pytest.raises(ValueError):
        _ = move_first_resets_to_beginning(circuit)

    circuit = stim.Circuit("I 0")

    with pytest.raises(ValueError):
        _ = move_first_resets_to_beginning(circuit)
    return


def test_remove_gauge_detectors():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        """
    )

    new_circuit = remove_gauge_detectors(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        DETECTOR(9) rec[-4] rec[-2]
        X 0
        CNOT 1 0
        """
    )

    # the DEM looks like "error(0.5) D0 D2"
    with pytest.raises(ValueError):
        _ = remove_gauge_detectors(circuit)

    return


def test_remove_detectors():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    new_circuit = remove_detectors(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        X 0
        CNOT 1 0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    new_circuit = remove_detectors(circuit, [0, 2, 1000])

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    assert new_circuit == expected_circuit

    with pytest.raises(TypeError):
        _ = remove_detectors(circuit, [1.2])

    return


def test_observables_to_detectors():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        R 0
        M 0
        OBSERVABLE_INCLUDE(1) rec[-1]
        """
    )

    new_circuit = observables_to_detectors(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        DETECTOR(0) rec[-1]
        R 0
        M 0
        DETECTOR(1) rec[-1]
        """
    )

    assert new_circuit == expected_circuit

    new_circuit = observables_to_detectors(circuit, observables=[1])

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        DETECTOR(9) rec[-4] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-1]
        R 0
        M 0
        DETECTOR(1) rec[-1]
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit("OBSERVABLE_INCLUDE(0) Z0")

    with pytest.raises(ValueError):
        _ = observables_to_detectors(circuit)

    circuit = stim.Circuit(
        """
        R 0 1
        M 0 1
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-2]
        """
    )

    with pytest.raises(ValueError):
        _ = observables_to_detectors(circuit)

    return


def test_move_observables_to_end():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        M 0 1
        MX 1
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-4]
        X 0
        CNOT 1 0
        M 0 1 3
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    new_circuit = move_observables_to_end(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        M 0 1
        MX 1
        X 0
        CNOT 1 0
        M 0 1 3
        DETECTOR(9) rec[-4] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-7] rec[-8]
        OBSERVABLE_INCLUDE(1) rec[-4] rec[-7]
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        X 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        X 0
        OBSERVABLE_INCLUDE(1) Z0
        """
    )

    new_circuit = move_observables_to_end(circuit)

    expected_circuit = stim.Circuit(
        """
        X 0
        M 0
        X 0
        OBSERVABLE_INCLUDE(1) Z0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        X 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        OBSERVABLE_INCLUDE(1) Z0
        X 0
        """
    )

    with pytest.raises(ValueError):
        _ = move_observables_to_end(circuit)

    return


def test_format_rec_targets():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        M 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        M 0 1
        MX 1
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-4]
        X 0
        CX 1 0
        M 0 1 3
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    circuit_str = format_rec_targets(circuit)

    expected_circuit_str = """R 0 1 2 3
X_ERROR(0.1) 0 1 2 3
MX 0
M 1 2 3
OBSERVABLE_INCLUDE(0) q3[-1] q2[-1]
DETECTOR(0) q0[-1]
DETECTOR(3) q1[-1] q3[-1]
M 0 1
MX 1
OBSERVABLE_INCLUDE(1) q1[-1] q3[-1]
X 0
CX 1 0
M 0 1 3
DETECTOR(9) q1[-2] q1[-1]
"""

    assert circuit_str == expected_circuit_str

    return


def test_format_to_rec_targets():
    circuit_str = """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        M 1 2 3
        OBSERVABLE_INCLUDE(0) q3[-1] q2[-1]
        DETECTOR(0) q0[-1]
        DETECTOR(3) q1[-1] q3[-1]
        M 0 1
        MX 1
        OBSERVABLE_INCLUDE(1) q1[-1] q3[-1]
        X 0
        CX 1 0
        M 0 1 3
        DETECTOR(9) q1[-2] q1[-1]
        """
    qubit_inds = {f"q{i}": i for i in range(4)}

    circuit = format_to_rec_targets(circuit_str, qubit_inds)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        M 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        M 0 1
        MX 1
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-4]
        X 0
        CX 1 0
        M 0 1 3
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    assert circuit == expected_circuit

    return


def test_formatting_rec_targets():
    qubit_inds = {"D1": 1, "D4": 2, "fdsjkjflkds": 3, "dfs": 0}
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        M 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
        DETECTOR(0) rec[-4]
        DETECTOR(3) rec[-3] rec[-1]
        M 0 1
        MX 1
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-4]
        X 0
        CX 1 0
        M 0 1 3
        DETECTOR(9) rec[-4] rec[-2]
        """
    )

    circuit_str = format_rec_targets(circuit, qubit_inds)
    new_circuit = format_to_rec_targets(circuit_str, qubit_inds)

    assert new_circuit == circuit

    return


def test_remove_non_native_instrs():
    circuit_str = """
R 0 1
M(0.1) 2
TICK
DETECTOR rec[-1] rec[-3]
S_DAG 2
LEAKAGE 1 0
LEAKAGE_NOISE(0.1) 1 0"""

    new_circuit_str = remove_non_native_instrs(circuit_str)

    expected_circuit_str = """R 0 1
M(0.1) 2
TICK
DETECTOR rec[-1] rec[-3]
S_DAG 2"""

    assert new_circuit_str == expected_circuit_str

    with pytest.raises(ValueError):
        _ = remove_non_native_instrs(circuit_str + "\nREPEAT 0 {")

    return


def test_remove_observables():
    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        OBSERVABLE_INCLUDE(9) rec[-4] rec[-2]
        """
    )

    new_circuit = remove_observables(circuit)

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        X 0
        CNOT 1 0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        OBSERVABLE_INCLUDE(9) rec[-4] rec[-2]
        """
    )

    new_circuit = remove_observables(circuit, [0, 3, 1000])

    expected_circuit = stim.Circuit(
        """
        R 0 1 2 3
        X_ERROR(0.1) 0 1 2 3
        MX 0
        MZ 1 2 3
        OBSERVABLE_INCLUDE(0) rec[-4]
        OBSERVABLE_INCLUDE(3) rec[-3] rec[-1]
        X 0
        CNOT 1 0
        """
    )

    assert new_circuit == expected_circuit

    with pytest.raises(TypeError):
        _ = remove_detectors(circuit, [1.2])

    return


def test_redefine_observables():
    circuit = stim.Circuit(
        """
        M 0 1 2
        OBSERVABLE_INCLUDE(0) Z0
        OBSERVABLE_INCLUDE(1) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(0) rec[-3]
        """
    )

    new_circuit = redefine_observables(circuit, {5: [0, 1], 6: [1], 7: [0]})

    expected_circuit = stim.Circuit(
        """
        M 0 1 2
        OBSERVABLE_INCLUDE(5) rec[-3] Z0 rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(6) rec[-1] rec[-2]
        OBSERVABLE_INCLUDE(7) rec[-3] Z0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        OBSERVABLE_INCLUDE(0) Z0
        X 0
        OBSERVABLE_INCLUDE(1) Z2
        """
    )

    with pytest.raises(ValueError):
        _ = redefine_observables(circuit, {0: [1, 0]})

    new_circuit = redefine_observables(circuit, {0: [1]})

    expected_circuit = stim.Circuit(
        """
        OBSERVABLE_INCLUDE(0) Z0
        X 0
        OBSERVABLE_INCLUDE(0) Z2
        """
    )

    assert new_circuit == expected_circuit

    return


def test_merge_observable_definitions():
    circuit = stim.Circuit(
        """
        M 0 1 2
        OBSERVABLE_INCLUDE(1) rec[-3]
        X 0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(1) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-1]
        X 0
        OBSERVABLE_INCLUDE(0) Z0 Z1
        M 0
        OBSERVABLE_INCLUDE(0) rec[-3]
        """
    )

    new_circuit = merge_observable_definitions(circuit)

    expected_circuit = stim.Circuit(
        """
        M 0 1 2
        X 0
        M 0
        X 0
        OBSERVABLE_INCLUDE(0) Z0 Z1 rec[-2] rec[-1] rec[-2]
        M 0
        OBSERVABLE_INCLUDE(1) rec[-5] rec[-2]
        OBSERVABLE_INCLUDE(2) rec[-2]
        """
    )

    assert new_circuit == expected_circuit

    new_circuit = merge_observable_definitions(circuit, observables=[0])

    expected_circuit = stim.Circuit(
        """
        M 0 1 2
        OBSERVABLE_INCLUDE(1) rec[-3]
        X 0
        M 0
        OBSERVABLE_INCLUDE(1) rec[-1]
        OBSERVABLE_INCLUDE(2) rec[-1]
        X 0
        OBSERVABLE_INCLUDE(0) Z0 Z1 rec[-2] rec[-1] rec[-2]
        M 0
        """
    )

    assert new_circuit == expected_circuit

    circuit = stim.Circuit(
        """
        OBSERVABLE_INCLUDE(0) Z0
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )

    with pytest.raises(ValueError):
        _ = merge_observable_definitions(circuit)

    return
