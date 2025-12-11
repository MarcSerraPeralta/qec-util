import xarray as xr

from qec_util.data_processing import syndrome


def test_syndrome():
    methods = [
        "get_syndromes",
        "get_defects",
        "get_final_defects",
        "get_defect_probs",
        "get_final_defect_probs",
    ]

    assert set(dir(syndrome)) >= set(methods)

    return


def test_get_syndromes():
    anc_meas = xr.DataArray(
        [[[0, 1, 0, 0, 0, 1, 1]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
            meas_reset=True,
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    syndromes = syndrome.get_syndromes(anc_meas)

    assert (syndromes == anc_meas).all()

    anc_meas["meas_reset"] = False
    syndromes = syndrome.get_syndromes(anc_meas)

    expected_syndromes = xr.DataArray(
        [[[0, 1, 1, 0, 0, 1, 0]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
            meas_reset=True,
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    assert (syndromes == expected_syndromes).all()

    return


def test_get_defects():
    syndromes = xr.DataArray(
        [[[0, 1, 0, 0, 0, 1, 1]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    defects = syndrome.get_defects(syndromes)

    expected_defects = xr.DataArray(
        [[[0, 1, 1, 0, 0, 1, 0]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    assert (defects == expected_defects).all()

    frame = xr.DataArray(
        [1],
        coords=dict(
            anc_qubit=["X1"],
        ),
        dims=("anc_qubit",),
    )

    defects = syndrome.get_defects(syndromes, frame=frame)

    expected_defects = xr.DataArray(
        [[[1, 1, 1, 0, 0, 1, 0]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    assert (defects == expected_defects).all()

    return


def test_get_final_defects():
    syndromes = xr.DataArray(
        [[[0, 1, 0, 0, 0, 1, 1]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )
    proj_syndromes = xr.DataArray(
        [[1]],
        coords=dict(anc_qubit=["X1"], shot=[1]),
        dims=("shot", "anc_qubit"),
    )

    final_defects = syndrome.get_final_defects(syndromes, proj_syndromes)

    expected_final_defects = xr.DataArray(
        [[0]],
        coords=dict(anc_qubit=["X1"], shot=[1]),
        dims=("shot", "anc_qubit"),
    )

    assert (final_defects == expected_final_defects).all()

    return


def test_get_defect_probs():
    anc_probs = xr.DataArray(
        [[[[1, 0], [0, 1], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            state=[0, 1],
            qec_round=list(range(1, 7 + 1)),
            meas_reset=True,
        ),
        dims=("shot", "anc_qubit", "qec_round", "state"),
    )
    ideal_defects = xr.DataArray(
        [[0, 0, 0, 0, 0, 0, 1]],
        coords=dict(
            anc_qubit=["X1"],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("anc_qubit", "qec_round"),
    )

    defect_probs = syndrome.get_defect_probs(anc_probs, ideal_defects)

    expected_defect_probs = xr.DataArray(
        [[[0, 1, 1, 0, 0, 1, 1]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    assert (defect_probs == expected_defect_probs).all()

    anc_probs = xr.DataArray(
        [[[[1, 0], [3 / 4, 1 / 4], [1, 0], [1, 0], [1, 0], [0, 1], [0, 1]]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
            state=[0, 1],
            meas_reset=False,
        ),
        dims=("shot", "anc_qubit", "qec_round", "state"),
    )
    ideal_defects = xr.DataArray(
        [[0, 0, 0, 0, 0, 0, 0]],
        coords=dict(
            anc_qubit=["X1"],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("anc_qubit", "qec_round"),
    )

    defect_probs = syndrome.get_defect_probs(anc_probs, ideal_defects)

    expected_defect_probs = xr.DataArray(
        [[[0, 1 / 4, 0, 1 / 4, 0, 1, 1]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            qec_round=list(range(1, 7 + 1)),
        ),
        dims=("shot", "anc_qubit", "qec_round"),
    )

    assert (defect_probs == expected_defect_probs).all()

    return


def test_get_final_defect_probs():
    anc_probs = xr.DataArray(
        [[[[0, 1]]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            state=[0, 1],
            qec_round=[7],
            meas_reset=True,
        ),
        dims=("shot", "anc_qubit", "qec_round", "state"),
    )
    data_probs = xr.DataArray(
        [[[0, 1], [0, 1]]],
        coords=dict(
            data_qubit=["D1", "D2"],
            shot=[1],
            state=[0, 1],
            meas_reset=True,
        ),
        dims=("shot", "data_qubit", "state"),
    )
    proj_matrix = xr.DataArray(
        [[1, 1]],
        coords=dict(data_qubit=["D1", "D2"], anc_qubit=["X1"]),
        dims=("anc_qubit", "data_qubit"),
    )
    ideal_final_defects = xr.DataArray(
        [1],
        coords=dict(anc_qubit=["X1"]),
        dims=("anc_qubit",),
    )

    final_defect_probs = syndrome.get_final_defect_probs(
        anc_probs, data_probs, ideal_final_defects, proj_matrix
    )

    expected_final_defect_probs = xr.DataArray(
        [[0]],
        coords=dict(anc_qubit=["X1"], shot=[1]),
        dims=("shot", "anc_qubit"),
    )

    assert (final_defect_probs == expected_final_defect_probs).all()

    anc_probs = xr.DataArray(
        [[[[3 / 4, 1 / 4], [0, 1]]]],
        coords=dict(
            anc_qubit=["X1"],
            shot=[1],
            state=[0, 1],
            qec_round=[6, 7],
            meas_reset=False,
        ),
        dims=("shot", "anc_qubit", "qec_round", "state"),
    )

    final_defect_probs = syndrome.get_final_defect_probs(
        anc_probs, data_probs, ideal_final_defects, proj_matrix
    )

    expected_final_defect_probs = xr.DataArray(
        [[1 / 4]],
        coords=dict(anc_qubit=["X1"], shot=[1]),
        dims=("shot", "anc_qubit"),
    )

    assert (final_defect_probs == expected_final_defect_probs).all()

    return
