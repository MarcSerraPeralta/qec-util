import pytest
import stim

from qec_util.dems import (
    remove_gauge_detectors,
    dem_difference,
    is_instr_in_dem,
    get_max_weight_hyperedge,
    disjoint_graphs,
    get_flippable_detectors,
    get_flippable_logicals,
    contains_only_edges,
    convert_observables_to_detectors,
    get_errors_triggering_detectors,
    only_errors,
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


def test_get_flippable_logicals():
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

    logs = get_flippable_logicals(dem)

    expected_logs = set([2, 3])

    assert logs == expected_logs

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


def test_convert_observables_to_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(1) D2 D3 L0
        error(0.1) D1 L1
        detector(1, 1.1) D0
        logical_observable L0
        """
    )

    new_dem = convert_observables_to_detectors(dem, [0], [123])

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

    new_dem = convert_observables_to_detectors(dem)

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
