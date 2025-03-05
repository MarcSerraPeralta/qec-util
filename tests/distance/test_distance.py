import stim

from qec_util.distance import get_circuit_distance, get_circuit_distance_logical
from qec_util.dem_instrs import get_logicals, get_detectors


def test_get_circuit_distance():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )

    d_circ = get_circuit_distance(circuit)

    assert d_circ == 3

    return


def test_get_circuit_distance_logical():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()

    d_circ, errors = get_circuit_distance_logical(dem, logical_id=0)

    assert d_circ == 3
    assert isinstance(errors, stim.DetectorErrorModel)
    assert len(errors) == d_circ

    for error in errors:
        assert error in dem

    dets, logs = set(), set()
    for error in errors:
        dets.symmetric_difference_update(get_detectors(error))
        logs.symmetric_difference_update(get_logicals(error))
    assert dets == set()
    assert logs != set()

    return
