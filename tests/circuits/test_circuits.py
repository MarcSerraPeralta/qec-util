import stim
import pytest

from qec_util.circuits import (
    remove_gauge_detectors,
    remove_detectors_except,
    observables_to_detectors,
    move_observables_to_end,
    format_rec_targets,
)


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


def test_remove_detectors_except():
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

    new_circuit = remove_detectors_except(circuit)

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

    new_circuit = remove_detectors_except(circuit, [0, 2, 1000])

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
        _ = remove_detectors_except(circuit, [1.2])

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
        """
    )

    assert new_circuit == expected_circuit

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
