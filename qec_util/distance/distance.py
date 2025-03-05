import numpy as np
import stim
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import gurobipy as gp
from gurobipy import GRB

from ..dems import (
    convert_logical_to_detector,
    only_errors,
    get_errors_triggering_detectors,
)


def get_circuit_distance(circuit: stim.Circuit) -> int:
    """Returns the circuit distance of the given circuit.

    Note that the SAT solver can take some time.

    Parameters
    ----------
    circuit
        Stim circuit.

    Returns
    -------
    d_circ
        Circuit distance of the given circuit.
    """
    if not isinstance(circuit, stim.Circuit):
        raise ValueError(
            "'circuit' must be a 'stim.Circuit', " f"but {type(circuit)} was given."
        )

    # remove gauge detectors from experiment (if not, it doesn't work)
    dem = circuit.detector_error_model(allow_gauge_detectors=True)
    gauge_dets = []
    for line in dem.flattened():
        if line.type == "error" and line.args_copy()[0] == 0.5:
            gauge_dets += line.targets_copy()
    gauge_dets = [d.val for d in gauge_dets]

    new_circuit = stim.Circuit()
    det_counter = -1
    for line in circuit.flattened():
        if line.name == "DETECTOR":
            det_counter += 1
            if det_counter in gauge_dets:
                continue

        new_circuit.append(line)

    # solve SAT problem
    wcnf_string = new_circuit.shortest_error_sat_problem()
    wcnf = WCNF(from_string=wcnf_string)
    with RC2(wcnf) as rc2:
        rc2.compute()
        d_circ = rc2.cost

    return d_circ


def get_circuit_distance_logical(
    dem: stim.DetectorErrorModel, logical_id: int
) -> tuple[int, stim.DetectorErrorModel]:
    """Returns the minimum number of faults to flip the specified logical
    without triggering any detectors given the detector error model.

    Parameters
    ----------
    dem
        Detector error model.
    logical_id
        Index of the logical observable in the ``dem``.

    Returns
    -------
    d_circ
        Circuit distance of the ``logical``.
    errors
        Set of faults that makes the circuit distance be ``d_circ``.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    dem = dem.flattened()
    dem = only_errors(dem)
    logical_det_id = dem.num_detectors
    new_dem = convert_logical_to_detector(
        dem, logical_id=logical_id, detector_id=logical_det_id
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
        defect = 1 if det_id == logical_det_id else 0
        support = np.array(support)

        model.addConstr(
            errors[support] @ np.ones_like(support) - 2 * dummy[det_id] == defect,
            f"syndrome{det_id}",
        )

    # define cost function to maximize
    obj_fn = 0
    for k in range(new_dem.num_errors):
        obj_fn += errors[k]
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
