import pytest
import stim

from qec_util.dems import decompose_hyperedges_to_edges, decomposed_graphlike_dem


def test_decompose_hyperedges_to_edges():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D4 D1 L0
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        detector(0, 2, 1) D0
        """
    )

    decom_dem = decompose_hyperedges_to_edges(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D7 L0 ^ D7 D4 L0 ^ D1 L0
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        detector(0, 2, 1) D0
        """
    )

    assert decom_dem == expected_dem

    return


def test_decompose_hyperedges_to_edges_decomposition_failure():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D4 D1
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        detector(0, 2, 1) D0
        """
    )

    decom_dem = decompose_hyperedges_to_edges(dem, ignore_decomposition_failures=True)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D7 L0 ^ D7 D4 L0 ^ D1 L0
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        detector(0, 2, 1) D0
        """
    )

    assert decom_dem == expected_dem

    with pytest.raises(ValueError):
        _ = decompose_hyperedges_to_edges(dem, ignore_decomposition_failures=False)

    return


def test_decomposed_graphlike_dem():
    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D4 ^ D0 D1 L0
        error(0.2) D0 D4
        error(0.2) D0 D1 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        error(0.2) D3 D4 L0
        error(0.1) D1 D2 ^ D3 D4 L0
        detector(0, 2, 1) D0
        logical_observable L1
        """
    )

    decom_dem = decomposed_graphlike_dem(dem)

    expected_dem = stim.DetectorErrorModel(
        """
        error(0.26) D0 D4
        error(0.26) D0 D1 L0
        error(0.2) D1 L0
        error(0.26) D1 D2
        error(0.26) D3 D4 L0
        detector(0, 2, 1) D0
        logical_observable L1
        """
    )

    assert decom_dem == expected_dem

    dem = stim.DetectorErrorModel(
        """
        error(0.1) D0 D4 D1 L0
        error(0.2) D0 D4
        error(0.2) D0 D1 L0
        """
    )

    with pytest.raises(ValueError):
        _ = decomposed_graphlike_dem(dem)

    return
