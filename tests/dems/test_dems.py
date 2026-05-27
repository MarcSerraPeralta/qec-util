import numpy as np
import pytest
import stim

from qec_util.dems import (
    contains_only_edges,
    dem_difference,
    detectors_to_observables,
    disjoint_graphs,
    get_errors_triggering_detectors,
    get_flippable_detectors,
    get_flippable_observables,
    get_max_weight_hyperedge,
    is_instr_in_dem,
    observables_to_detectors,
    only_errors,
    prepare_distance2_dem_for_pymatching,
    remove_fake_errors,
    remove_gauge_detectors,
    remove_hyperedges,
    separate_edges_and_hyperedges,
)


def test_remove_gauge_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.5) D4 
        error(0.2) D1 D2
        """
    )

    new_dem = remove_gauge_detectors(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.2) D1 D2
        """
    )

    assert new_dem == expected_dem

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.5) D1 D2 D3
        error(0.2) D1 D2
        """
    )
    with pytest.raises(ValueError):
        _ = remove_gauge_detectors(dem)

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.5) D1
        error(0.2) D1 D2
        """
    )
    with pytest.raises(ValueError):
        _ = remove_gauge_detectors(dem)

    return


def test_dem_difference():
    dem_1 = stim.DetectorErrorModel(
        """
        error(0.1) L0 D0
        error(0.2) D1 ^ D2
        error(0.3) D3 D4 D1
        """
    )
    dem_2 = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.2) D1 D2
        error(0.3) D1 D3 D4
        """
    )

    diff_1, diff_2 = dem_difference(dem_1, dem_2)

    assert len(diff_1) == 0
    assert len(diff_2) == 0

    dem_2 = stim.DetectorErrorModel(
        """
        error(0.2) D1 D2
        error(0.3) D1 D3 D4
        error(0.5) D0
        """
    )

    diff_1, diff_2 = dem_difference(dem_1, dem_2)

    assert diff_1 == stim.DetectorErrorModel("error(0.1) D0 L0")
    assert diff_2 == stim.DetectorErrorModel("error(0.5) D0")

    return


def test_is_instr_in_dem():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) L0 D0
        error(0.2) D1 ^ D2
        error(0.3) D3 D4 D1
        error(0.5) D1 L1
        """
    )
    dem_instr = stim.DemInstruction(
        "error",
        [0.1],
        [stim.target_relative_detector_id(0), stim.target_logical_observable_id(0)],
    )
    assert is_instr_in_dem(dem_instr, dem)

    dem_instr = stim.DemInstruction(
        "error",
        [0.2],
        [stim.target_relative_detector_id(0), stim.target_logical_observable_id(0)],
    )
    assert not is_instr_in_dem(dem_instr, dem)

    dem_instr = stim.DemInstruction(
        "error",
        [0.5],
        [stim.target_relative_detector_id(1), stim.target_logical_observable_id(1)],
    )
    assert is_instr_in_dem(dem_instr, dem)

    dem_instr = stim.DemInstruction(
        "detector",
        [0],
        [stim.target_relative_detector_id(1)],
    )
    with pytest.raises(TypeError):
        is_instr_in_dem(dem_instr, dem)

    return


def test_get_max_weight_hyperedge():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) L0 D0
        error(0.2) D1 ^ D2
        error(0.3) D3 D4 D1
        error(0.5) D1 L1
        """
    )

    max_weight, hyperedge = get_max_weight_hyperedge(dem)

    expected_hyperedge = stim.DemInstruction(
        "error",
        args=[0.3],
        targets=[
            stim.target_relative_detector_id(3),
            stim.target_relative_detector_id(4),
            stim.target_relative_detector_id(1),
        ],
    )

    assert max_weight == 3
    assert hyperedge == expected_hyperedge

    dem = stim.DetectorErrorModel()

    max_weight, hyperedge = get_max_weight_hyperedge(dem)

    expected_hyperedge = stim.DemInstruction(
        "error",
        args=[0.0],
        targets=[],
    )

    assert max_weight == 0
    assert hyperedge == expected_hyperedge

    return


def test_disjoint_graphs():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) L0 D0
        error(0.2) D1 ^ D2
        error(0.3) D3 D4 D1
        error(0.5) D5 D6
        detector(0) D0
        """
    )

    subgraphs = disjoint_graphs(dem)
    subgraphs = set(tuple(sorted(s)) for s in subgraphs)

    expected_subgraphs = set([(0,), (1, 2, 3, 4), (5, 6)])

    assert subgraphs == expected_subgraphs

    return


def test_get_flippable_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D4 D6
        error(0.3) D3 L5
        detector D0
        detector D1
        detector D2
        detector D5
        """
    )

    dets = get_flippable_detectors(dem)

    expected_dets = set([3, 4, 6])

    assert dets == expected_dets

    return


def test_get_flippable_observables():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D4 D6
        error(0.3) D3 L2 L3
        detector D0
        detector D1
        detector D2
        detector D5
        """
    )

    obs = get_flippable_observables(dem)

    expected_obs = set([2, 3])

    assert obs == expected_obs

    return


def test_contains_only_edges():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1
        """
    )

    assert contains_only_edges(dem)

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D2
        error(1) D2 D3 D4
        """
    )

    assert not contains_only_edges(dem)

    return


def test_observables_to_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    new_dem = observables_to_detectors(dem, [0], [123])

    expected_dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 D123
        error(0.1) D1 L1
        detector(1, 1.1) D0
        detector D123
        """
    )

    assert new_dem == expected_dem

    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        """
    )

    new_dem = observables_to_detectors(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 D4
        error(0.1) D1 D5
        detector(1, 1.1) D0
        """
    )

    assert new_dem == expected_dem

    return


def test_get_errors_triggering_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        error(0.2) D2
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    support = get_errors_triggering_detectors(dem, detectors=[1, 2, 3])

    expected_support = {
        1: [1],
        2: [0, 2],
        3: [0],
    }

    assert support == expected_support

    return


def test_only_errors():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    new_dem = only_errors(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        """
    )

    assert new_dem == expected_dem

    return


def test_remove_hyperedges():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        error(0.1) D1 D2 D3 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    new_dem = remove_hyperedges(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    assert new_dem == expected_dem

    return


def test_separate_edges_and_hyperedges():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        error(0.1) D1 D2 D3 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    graph_dem, hyper_dem = separate_edges_and_hyperedges(dem)

    expected_graph_dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )
    expected_hyper_dem = stim.DetectorErrorModel(
        """
        error(0.1) D1 D2 D3 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    assert graph_dem == expected_graph_dem
    assert hyper_dem == expected_hyper_dem

    return


def test_prepare_distance2_dem_for_pymatching():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1
        error(0.1) D0 L0
        error(0.2) D0
        error(0.4) D3 D5
        logical_observable L0
        """
    )

    new_dem = prepare_distance2_dem_for_pymatching(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D1
        error(0.2) D0
        error(0.4) D3 D5
        logical_observable L0
        """
    )

    assert new_dem == expected_dem

    return


def test_remove_fake_errors():
    circuit = stim.Circuit(
        """
        R 0
        Z_ERROR(0.1) 0
        MX 0
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.1) D0")

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        DEPOLARIZE1(0.1) 0
        MX 0
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.06666666666666667) D0")

    assert len(new_dem) == len(expected_dem)
    assert new_dem[0].targets_copy() == expected_dem[0].targets_copy()
    assert np.isclose(new_dem[0].args_copy()[0], expected_dem[0].args_copy()[0])

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        MX(0.1) 0
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.1) D0")

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        X_ERROR(0.1) 0
        MX 0
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("")

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        RX 1
        TICK
        CNOT 1 0
        TICK
        X_ERROR(0.1) 0
        TICK
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.1) D0")

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        RX 1
        TICK
        CNOT 1 0
        TICK
        X_ERROR(0.1) 0
        TICK
        M 0 1
        DETECTOR rec[-2] rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.1) D0")

    assert new_dem == expected_dem

    circuit = stim.Circuit(
        """
        R 0
        RX 1
        X_ERROR(0.1) 1
        TICK
        CNOT 1 0
        TICK
        TICK
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
        """
    )
    dem = circuit.detector_error_model(allow_gauge_detectors=True)

    new_dem = remove_fake_errors(circuit, dem)

    expected_dem = stim.DetectorErrorModel("error(0.1) D0 D1")

    assert new_dem == expected_dem

    return


def test_detectors_to_observables():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 ^ D0 L0 ^ L0
        error(0.2) D1 D2 L2
        error(0.2) D1 D4 L2
        detector(1, 1, 2) D100
        detector(1, 1, 3) D0
        """
    )
    det_to_obs = {
        stim.target_relative_detector_id(0): stim.target_logical_observable_id(1),
        stim.target_relative_detector_id(2): stim.target_logical_observable_id(3),
    }

    new_dem = detectors_to_observables(dem, det_to_obs)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) L1 ^ L1 L0 ^ L0
        error(0.2) D1 L3 L2
        error(0.2) D1 D4 L2
        detector(1, 1, 2) D100
        """
    )

    assert new_dem == expected_dem

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 ^ D0 L0 ^ L0
        error(0.2) D1 D2 L2
        error(0.2) D1 D4 L2
        detector(1, 1, 3) D0
        """
    )

    new_dem = detectors_to_observables(dem, 1)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 ^ D0 L0 ^ L0
        error(0.2) D1 D2 L2
        error(0.2) D1 L0 L2
        detector(1, 1, 3) D0
        """
    )

    assert new_dem == expected_dem

    return
