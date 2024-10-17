import stim

from qec_util.dem_instrs import (
    get_detectors,
    get_logicals,
    has_separator,
    sorted_dem_instr,
    remove_detectors,
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


def test_get_logicals():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 L0 ^ D1 L1 ^ D0 L1
        error(0.1) D0 D1 L2 L1
        """
    )

    logicals = get_logicals(dem[0])

    expected_logicals = (0,)
    assert logicals == expected_logicals

    logicals = get_logicals(dem[1])

    expected_logicals = (1, 2)
    assert logicals == expected_logicals

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
        error(0.1) D0 L0 ^ D1 L1 ^ D0
        error(0.1) D0 D1 L2 L1
        """
    )

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) L0 ^ D1 L1
        error(0.1) D1 L2 L1
        """
    )

    for instr, exp_instr in zip(dem, expected_dem):
        output = remove_detectors(instr, dets=[0])
        assert output == exp_instr

    return


def test_remove_detectors():
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

    return
