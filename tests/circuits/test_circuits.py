import stim
import pytest

from qec_util.circuits import (
    remove_gauge_detectors,
    remove_detectors_except,
    logicals_to_detectors,
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


def test_logicals_to_detectors():
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

    new_circuit = logicals_to_detectors(circuit)

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
