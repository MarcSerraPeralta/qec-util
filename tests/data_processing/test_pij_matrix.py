import stim
import matplotlib.pyplot as plt

from qec_util.data_processing import (
    get_pij_matrix,
    get_approx_pij_matrix,
    plot_pij_matrix,
)


def test_pij_matrix(show_figures):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=5,
        after_clifford_depolarization=0.05,
    )
    sampler = circuit.compile_detector_sampler()
    defects = sampler.sample(shots=5_000)

    pij = get_pij_matrix(defects)

    assert pij.shape == (2 * 6, 2 * 6)

    pij_approx = get_approx_pij_matrix(defects)

    assert pij_approx.shape == (2 * 6, 2 * 6)
    assert (pij_approx > pij).all()
    assert (pij_approx - pij < 0.1).all()

    _, ax = plt.subplots()

    plot_pij_matrix(ax, pij, qubit_labels=["A1", "A2"], num_rounds=5 + 1, max_prob=0.03)

    if show_figures:
        plt.show()
    plt.close()

    return
