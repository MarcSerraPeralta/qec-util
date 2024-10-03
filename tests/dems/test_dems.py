import pytest
import stim

from qec_util.dems import remove_gauge_detectors, dem_difference, is_instr_in_dem


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
