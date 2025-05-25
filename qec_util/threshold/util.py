from collections.abc import Callable

import pathlib
import numpy as np


def get_fit_func(fit_func: str) -> tuple[Callable, int]:
    if fit_func == "tanh":

        def tanh(x, a, b, c):
            return a * (1 - (1 - 0.5 * (1 + np.tanh(b * x))) ** c)

        return tanh, 3

    elif len(fit_func) >= 5 and fit_func[:5] == "poly-":
        order = fit_func[5:]
        try:
            order = int(order)
        except ValueError:
            raise ValueError(f"'{fit_func}' is not a valid name for 'poly' fit_func.")

        def poly(x, *args):
            return sum([args[i] * x**i for i in range(order + 1)])

        return poly, order + 1

    else:
        raise ValueError(f"'{fit_func}' is not a valid name for 'fit_func'.")


def rescale_input(
    phys_err: np.ndarray | int | float,
    distance: np.ndarray | int | float,
    p_threshold: np.ndarray | float,
    mu: np.ndarray | int | float,
) -> np.ndarray:
    return (phys_err - p_threshold) * distance ** (1 / mu)


def save_fit_information(
    file_name: str | pathlib.Path,
    fit_func_name: str,
    popt: np.ndarray,
    pcov: np.ndarray,
    bootstrap_thresholds: np.ndarray | None = None,
) -> None:
    """
    Stores the fit information to the given file name for a YAML.

    This information is stored when running ``qec_util.threshold.get_threshold``
    if given the ``file_name`` parameter.
    The fit information is stored as a dictionary with the same keys as
    the arguments in this function.

    Notes
    -----
    This function requires ``pyyaml``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    import yaml

    # ensure that they are np.ndarrays
    popt, pcov = np.array(popt), np.array(pcov)
    bootstrap_thresholds = np.array(bootstrap_thresholds)

    data = {
        "fit_func_name": fit_func_name,
        "param_names": ["p_threshold", "mu"]
        + [f"fit_func_param_{i+1}" for i, _ in enumerate(popt[2:])],
        "popt": popt.tolist(),
        "pcov": pcov.tolist(),
        "bootstrap_thresholds": bootstrap_thresholds.tolist(),
    }

    with open(file_name, "w") as file:
        yaml.dump(data, file, default_flow_style=False)

    return


def load_fit_information(
    file_name: str | pathlib.Path,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Returns 'fit_func_name', 'popt', 'pcov' and 'bootstrap_thresholds'.
    See 'save_fit_information' for more information.

    Notes
    -----
    This function requires ``pyyaml``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    import yaml

    with open(file_name, "r") as file:
        data = yaml.safe_load(file)

    samples_thr = data["bootstrap_thresholds"]
    samples_thr = np.array(samples_thr) if samples_thr is not None else samples_thr
    return (
        data["fit_func_name"],
        np.array(data["popt"]),
        np.array(data["pcov"]),
        samples_thr,
    )
