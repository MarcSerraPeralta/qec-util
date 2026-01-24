import matplotlib.pyplot as plt

from qec_util.threshold import (
    plot_threshold_fit,
    plot_threshold_data,
    get_threshold,
    load_fit_information,
)


def test_plot_threshold(show_figures, tmp_path):
    p = [0, 0.1, 0.2, 0.3]
    data = {
        3: (p, [0, 10, 20, 30], [1000, 234, 234, 235]),
        5: (p, [0, 4, 25, 45], [2000, 234, 234, 240]),
        7: (p, [0, 1, 36, 60], [1234, 234, 234, 241]),
    }
    get_threshold(data, file_name=tmp_path / "tmp_plot.yaml")
    fit_func_name, popt, _, thr_samples = load_fit_information(
        tmp_path / "tmp_plot.yaml"
    )

    _, ax = plt.subplots()
    plot_threshold_data(ax, data, popt)
    plot_threshold_fit(ax, fit_func_name, popt)

    if show_figures:
        plt.show()
    plt.close()

    return
