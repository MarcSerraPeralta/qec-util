import stim
import matplotlib.pyplot as plt

from qec_util.data_processing import get_pij_matrix, plot_pij_matrix


def test_pij_matrix(show_figures):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=5,
        after_clifford_depolarization=0.10,
    )
    sampler = circuit.compile_detector_sampler()
    defects = sampler.sample(shots=1_000)

    pij = get_pij_matrix(defects)

    assert pij.shape == (2 * 6, 2 * 6)

    _, ax = plt.subplots()

    plot_pij_matrix(ax, pij, qubit_labels=["A1", "A2"], num_rounds=5 + 1)

    if show_figures:
        plt.show()
    plt.close()

    return
