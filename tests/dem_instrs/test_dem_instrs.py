import pytest
import stim

from qec_util.dem_instrs import (
    decomposed_instrs,
    get_detectors,
    get_labels_from_detectors,
    get_observables,
    has_separator,
    merge_instrs,
    remove_detectors,
    sorted_dem_instr,
)


def test_get_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1 ^ D0 L1
        error(0.1) D0 D1 L2 L1
        """
    )

    detectors = get_detectors(dem[0])

    expected_detectors = (1,)
    assert detectors == expected_detectors

    detectors = get_detectors(dem[1])

    expected_detectors = (0, 1)
    assert detectors == expected_detectors

    return


def test_get_observables():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1 ^ D0 L1
        error(0.1) D0 D1 L2 L1
        """
    )

    obs = get_observables(dem[0])

    expected_obs = (0,)
    assert obs == expected_obs

    obs = get_observables(dem[1])

    expected_obs = (1, 2)
    assert obs == expected_obs

    return


def test_has_separator():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1 ^ D0 L1
        error(0.1) D0 D1 L2 L1
        """
    )

    assert has_separator(dem[0])
    assert not has_separator(dem[1])

    return


def test_remove_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 ^ D0 ^ D0
        error(0.1) D0 L0 ^ D1 L1 ^ D0
        error(0.1) D0 D1 L2 L1
        """
    )

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1)
        error(0.1) L0 ^ D1 L1
        error(0.1) D1 L2 L1
        """
    )

    for instr, exp_instr in zip(dem, expected_dem):
        output = remove_detectors(instr, dets=[0])
        assert output == exp_instr

    return


def test_sorted_dem_instr():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1 ^ D0
        error(0.1) L2 D1 D0 L1
        """
    )

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D1 L0 L1
        error(0.1) D0 D1 L1 L2
        """
    )

    for instr, exp_instr in zip(dem, expected_dem):
        output = sorted_dem_instr(instr)
        assert output == exp_instr

    expected_dem = stim.DetectorErrorModel(
        """
        error(0) D1 L0 L1
        error(0) D0 D1 L1 L2
        """
    )

    for instr, exp_instr in zip(dem, expected_dem):
        output = sorted_dem_instr(instr, prob=0)
        assert output == exp_instr

    return


def test_get_labels_from_detectors():
    anc_coords = {"X2": [1, 0.0]}
    det_coords = {13: [1, 0.0, 3]}
    assert get_labels_from_detectors(
        [13], det_coords=det_coords, anc_coords=anc_coords
    ) == [("X2", 3)]
    return


def test_decomposed_instrs():
    dem_instr = stim.DetectorErrorModel("error(0.1) D0 L0 ^ D1 L1 ^ D0")[0]

    new_dem = decomposed_instrs(dem_instr)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0
        error(0.1) D1 L1
        error(0.1) D0
        """
    )

    assert new_dem == expected_dem

    new_dem = decomposed_instrs(dem_instr, prob_method="zero")

    expected_dem = stim.DetectorErrorModel(
        """
        error(0) D0 L0
        error(0) D1 L1
        error(0) D0
        """
    )

    assert new_dem == expected_dem

    return


def test_merge_instrs():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 ^ D1 D2 L0
        error(0.2) D2 D1 L0 ^ D0
        error(0.1) D0
        """
    )

    new_instr = merge_instrs(dem[0], dem[1])

    expected_instr = stim.DetectorErrorModel("error(0.26) D0 D1 D2 L0")[0]

    assert new_instr == expected_instr

    _ = merge_instrs(dem[0], dem[0], dem[0])

    with pytest.raises(ValueError):
        _ = merge_instrs(dem[0], dem[2])

    return
