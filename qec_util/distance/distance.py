from collections.abc import Sequence
from itertools import product
import numpy as np
import stim

from ..dems import (
    convert_observables_to_detectors,
    only_errors,
    get_errors_triggering_detectors,
)


class Decoder:
    def __init__(self, dem: stim.DetectorErrorModel, **kargs) -> None: ...
    def decode_to_faults_array(self, syndrome: np.ndarray) -> np.ndarray: ...


def get_circuit_distance(circuit: stim.Circuit) -> tuple[int, stim.DetectorErrorModel]:
    """Returns the circuit distance of the given circuit.

    Parameters
    ----------
    circuit
        Stim circuit.

    Returns
    -------
    d_circ
        Circuit distance of the given circuit.
    errors
        Set of faults that makes the circuit distance be ``d_circ``.

    Notes
    -----
    This function requires ``gurobipy``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    See ``README.md`` for how to set up the Gurobi license.
    """
    if not isinstance(circuit, stim.Circuit):
        raise ValueError(
            "'circuit' must be a 'stim.Circuit', " f"but {type(circuit)} was given."
        )

    dem = circuit.detector_error_model(allow_gauge_detectors=True)
    obs_inds = list(range(dem.num_observables))

    return get_circuit_distance_observable(dem, obs_inds=obs_inds)


def get_circuit_distance_observable(
    dem: stim.DetectorErrorModel, obs_inds: int | Sequence[int]
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns the minimum number of faults to flip the specified logical
    observable without triggering any detectors given the detector error model.

    Parameters
    ----------
    dem
        Detector error model.
    obs_inds
        Index(s) of the logical observable in the ``dem``.

    Returns
    -------
    d_circ
        Circuit distance of the ``obs_ind``.
    errors
        Set of faults that makes the circuit distance be ``d_circ``.

    Notes
    -----
    This function requires ``gurobipy``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    See ``README.md`` for how to set up the Gurobi license.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if isinstance(obs_inds, int):
        obs_inds = [obs_inds]
    if not isinstance(obs_inds, Sequence):
        raise TypeError(
            f"'obs_inds' must be an int or a list[int], but {type(obs_inds)} was given."
        )

    import gurobipy as gp
    from gurobipy import GRB

    dem = dem.flattened()
    dem = only_errors(dem)
    obs_det_inds = [dem.num_detectors + i for i, _ in enumerate(obs_inds)]
    new_dem = convert_observables_to_detectors(
        dem, obs_inds=obs_inds, det_inds=obs_det_inds
    )
    det_support = get_errors_triggering_detectors(new_dem)

    # define model
    model = gp.Model("milp")
    model.Params.OutputFlag = 0
    model.Params.LogToConsole = 0

    # define variables
    errors = model.addMVar(shape=new_dem.num_errors, vtype=GRB.BINARY, name="errors")
    dummy = model.addMVar(
        shape=new_dem.num_detectors, vtype=GRB.INTEGER, name="dummy", lb=0
    )

    # add constraints
    for det_id, support in det_support.items():
        if len(support) == 0:
            continue
        defect = 1 if det_id in obs_det_inds else 0
        support = np.array(support)

        model.addConstr(
            errors[support] @ np.ones_like(support) - 2 * dummy[det_id] == defect,
            f"syndrome{det_id}",
        )

    # define cost function to maximize
    obj_fn = np.ones(new_dem.num_errors).T @ errors
    model.setObjective(obj_fn, GRB.MINIMIZE)

    # update model to build the contraints and cost function
    model.update()

    # solve MILP problem
    model.optimize()

    # convert errors to stim.DetectorErrorModel (attribute 'x' has the numpy values)
    error_vars = []
    for k in range(new_dem.num_errors):
        error_vars.append(model.getVarByName(f"errors[{k}]"))
    error_ids = [k for k, v in enumerate(model.getAttr("X", error_vars)) if v]

    d_circ = len(error_ids)
    errors = stim.DetectorErrorModel()
    for error_id in error_ids:
        errors.append(dem[error_id])

    return d_circ, errors


def get_upper_bound_circuit_distance(
    dem: stim.DetectorErrorModel, decoder: type[Decoder] | None = None, **kargs
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns an upper bound for the circuit distance.

    Parameters
    ----------
    dem
        Detector error model.
    decoder
        Decoder used to obtain the upper bound. The inputs for the initialization
        must be ``dem`` and ``**kargs``. It must have the ``decode_to_faults_array``
        which must return an array of the predicted faults/errors for the given syndrome.
        By default ``None``, which loads BPOSD.
    **kargs
        Extra arguments for the initialization of the decoder with ``dem``.

    Returns
    -------
    int
        Upper bound for the circuit distance.
    error
        Error corresponding to the upper bound.

    Notes
    -----
    This function requires ``dem_decoders`` if ``decoder = None``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    if kargs == {} and (decoder is None):
        from dem_decoders import BP_OSD as decoder

        # default arguments for BP_OSD
        kargs = {
            "max_iter": 50,
            "bp_method": "ps",
            "osd_order": 5,
            "osd_method": "osd_cs",
        }

    num_obs = dem.num_observables
    if num_obs == 0:
        raise ValueError("'dem' does not contain any logical observable.")

    dem = dem.flattened()
    new_dem = convert_observables_to_detectors(dem)
    decoder_dem = decoder(new_dem, **kargs)

    num_faults = np.inf
    for comb in product([False, True], repeat=num_obs):
        if not any(comb):
            # skip [0,0,0...] case
            continue

        syndrome = np.zeros(new_dem.num_detectors, dtype=bool)
        syndrome[-num_obs:] = np.array(comb, dtype=bool)

        faults = decoder_dem.decode_to_faults_array(syndrome)
        if np.sum(faults) > num_faults:
            continue

        fault_inds = np.where(faults)[0]
        error = stim.DetectorErrorModel()
        for fault_ind in fault_inds:
            error += dem[fault_ind : fault_ind + 1]  # object is a DetectorErrorModel
        num_faults = len(error)

    return num_faults, error
