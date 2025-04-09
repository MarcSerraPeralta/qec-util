import pytest
import stim

from qec_util.dem_instrs import decompose_hyperedge_to_edges


def test_decompose_hyperedge_to_edges():
    hyperedge = stim.DetectorErrorModel("error(0.1) D0 D1 D4 L0")[0]
    dem_edges = stim.DetectorErrorModel(
        """
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        """
    )

    decom_hyperedge = decompose_hyperedge_to_edges(hyperedge, dem_edges)

    expected_decom = stim.DetectorErrorModel("error(0.1) D0 D7 L0 ^ D7 D4 L0 ^ D1 L0")
    expected_decom = expected_decom[0]

    assert decom_hyperedge == expected_decom

    return


def test_decompose_hyperedge_to_edges_decomposition_failure():
    hyperedge = stim.DetectorErrorModel("error(0.1) D0 D1 D4")[0]
    dem_edges = stim.DetectorErrorModel(
        """
        error(0.2) D0 D7 L0
        error(0.2) D7 D4 L0
        error(0.2) D1 L0
        error(0.2) D1 D2
        """
    )

    decom_hyperedge = decompose_hyperedge_to_edges(
        hyperedge, dem_edges, ignore_decomposition_failure=True
    )

    expected_decom = stim.DetectorErrorModel("error(0.1) D0 D7 L0 ^ D7 D4 L0 ^ D1 L0")
    expected_decom = expected_decom[0]

    assert decom_hyperedge == expected_decom

    with pytest.raises(ValueError):
        _ = decompose_hyperedge_to_edges(
            hyperedge, dem_edges, ignore_decomposition_failure=False
        )

    dem_edges = stim.DetectorErrorModel(
        """
        error(0.2) D1 L0
        error(0.2) D1 D2
        """
    )

    with pytest.raises(ValueError):
        _ = decompose_hyperedge_to_edges(
            hyperedge, dem_edges, ignore_decomposition_failure=False
        )

    return
