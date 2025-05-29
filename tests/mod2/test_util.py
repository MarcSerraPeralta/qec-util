import pytest
import galois
import numpy as np

from qec_util.mod2 import decompose_into_basis


def test_decompose_into_basis():
    a = galois.GF2(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 1, 1],
            ]
        )
    ).T
    b = galois.GF2(np.array([1, 1, 1, 1]))

    x = decompose_into_basis(vector=b, basis=a)

    assert isinstance(x, galois.Array)
    assert (a @ x == b).all()

    b = galois.GF2(np.array([1, 0, 0, 0]))
    with pytest.raises(ValueError):
        _ = decompose_into_basis(vector=b, basis=a)

    return


def test_decompose_into_basis_from_labels():
    vector = ["D1", "D2", "D3", "D4"]
    basis = [["D1", "D2"], ["D2", "D3"], ["D3", "D4"]]

    x = decompose_into_basis(vector=vector, basis=basis)

    assert set(x) == set([0, 2])

    vector = ["D1", "D2", "D3", "D4"]
    basis = {"Z1": ["D1", "D2"], "Z2": ["D2", "D3"], "Z3": ["D3", "D4"]}

    x = decompose_into_basis(vector=vector, basis=basis)

    assert set(x) == set(["Z1", "Z3"])

    with pytest.raises(TypeError):
        _ = decompose_into_basis(vector=np.zeros(4, dtype=int), basis=basis)

    return
