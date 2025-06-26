import stim

from qec_util.distance import (
    get_circuit_distance,
    get_circuit_distance_observable,
    get_upper_bound_circuit_distance,
)
from qec_util.dem_instrs import get_observables, get_detectors


def test_get_circuit_distance():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )

    d_circ, error = get_circuit_distance(circuit)

    assert d_circ == 3
    assert isinstance(error, stim.DetectorErrorModel)
    assert len(error) == 3

    return


def test_get_circuit_distance_observable():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()

    d_circ, errors = get_circuit_distance_observable(dem, obs_inds=0)

    assert d_circ == 3
    assert isinstance(errors, stim.DetectorErrorModel)
    assert len(errors) == d_circ

    for error in errors:
        assert error in dem

    dets, obs = set(), set()
    for error in errors:
        dets.symmetric_difference_update(get_detectors(error))
        obs.symmetric_difference_update(get_observables(error))
    assert dets == set()
    assert obs != set()

    return


def test_get_upper_bound_circuit_distance():
    circuit = stim.Circuit.generated(
        code_task="surface_code:rotated_memory_z",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()

    d_circ, error = get_upper_bound_circuit_distance(dem)

    assert d_circ == 3

    syndrome, obs = set(), set()
    for fault in error:
        syndrome.symmetric_difference_update(get_detectors(fault))
        obs.symmetric_difference_update(get_observables(fault))
    assert len(syndrome) == 0
    assert len(obs) != 0

    return
