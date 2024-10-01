import stim

from qec_util.dems import remove_gauge_detectors


def test_remove_gauge_detectors():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0
        error(0.5) D1 D2 D3
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

    return
