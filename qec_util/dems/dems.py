from collections.abc import Sequence

import stim

from ..dem_instrs import get_detectors, get_observables, sorted_dem_instr


def remove_gauge_detectors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Remove the gauge detectors from a DEM."""
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(f"'dem' is not a stim.DetectorErrorModel, but a {type(dem)}.")

    new_dem = stim.DetectorErrorModel()
    gauge_dets = set()

    for dem_instr in dem.flattened():
        if dem_instr.type != "error":
            new_dem.append(dem_instr)

        if dem_instr.args_copy() == [0.5]:
            det = dem_instr.targets_copy()
            if len(det) != 1:
                raise ValueError("There exist 'composed' gauge detector: {dem_instr}.")
            gauge_dets.add(det[0])
            continue

        if dem_instr.args_copy() != [0.5]:
            if len([i for i in dem_instr.targets_copy() if i in gauge_dets]) != 0:
                raise ValueError(
                    "A gauge detector is present in the following error:\n"
                    f"{dem_instr}\nGauge detectors = {gauge_dets}"
                )
            new_dem.append(dem_instr)

    return new_dem


def dem_difference(
    dem_1: stim.DetectorErrorModel, dem_2: stim.DetectorErrorModel
) -> tuple[stim.DetectorErrorModel, stim.DetectorErrorModel]:
    """Returns the the DEM error instructions in the first DEM that are not present
    in the second DEM and vice versa. Note that this does not take into account
    the decomposition of errors.

    Parameters
    ----------
    dem_1
        First detector error model.
    dem_2
        Second detector error model.

    Returns
    -------
    diff_1
        DEM instructions present in ``dem_1`` that are not present in ``dem_2``.
    diff_2
        DEM instructions present in ``dem_2`` that are not present in ``dem_1``.
    """
    if not isinstance(dem_1, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem_1' is not a stim.DetectorErrorModel, but a {type(dem_1)}."
        )
    if not isinstance(dem_2, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem_2' is not a stim.DetectorErrorModel, but a {type(dem_2)}."
        )

    dem_1_ordered = stim.DetectorErrorModel()
    for dem_instr in dem_1.flattened():
        if dem_instr.type != "error":
            continue

        dem_1_ordered.append(sorted_dem_instr(dem_instr))

    dem_2_ordered = stim.DetectorErrorModel()
    for dem_instr in dem_2.flattened():
        if dem_instr.type != "error":
            continue

        dem_2_ordered.append(sorted_dem_instr(dem_instr))

    diff_1 = stim.DetectorErrorModel()
    for dem_instr in dem_1_ordered:
        if dem_instr not in dem_2_ordered:
            diff_1.append(dem_instr)

    diff_2 = stim.DetectorErrorModel()
    for dem_instr in dem_2_ordered:
        if dem_instr not in dem_1_ordered:
            diff_2.append(dem_instr)

    return diff_1, diff_2


def is_instr_in_dem(
    dem_instr: stim.DemInstruction, dem: stim.DetectorErrorModel
) -> bool:
    """Checks if the DEM error instruction and its undecomposed form are present
    in the given DEM.
    """
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise TypeError(f"'dem_instr' is not an error, but a {dem_instr.type}.")
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    dem_instr = sorted_dem_instr(dem_instr)

    for other_instr in dem.flattened():
        if other_instr.type != "error":
            continue

        if dem_instr == sorted_dem_instr(other_instr):
            return True

    return False


def get_max_weight_hyperedge(
    dem: stim.DetectorErrorModel,
) -> tuple[int, stim.DemInstruction]:
    """Return the weight and hyperedges corresponding to the max-weight hyperedge.

    Parameters
    ----------
    dem
        Stim detector error model.

    Returns
    -------
    weight
        Weight of the max-weight hyperedge in ``dem``.
    hyperedge
        Hyperedge with the max-weight in ``dem``.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    max_weight = 0
    hyperedge = stim.DemInstruction(type="error", args=[0], targets=[])
    for dem_instr in dem.flattened():
        if dem_instr.type != "error":
            continue

        targets = dem_instr.targets_copy()
        targets = [t for t in targets if t.is_relative_detector_id()]
        if len(targets) > max_weight:
            max_weight = len(targets)
            hyperedge = dem_instr

    return max_weight, hyperedge


def disjoint_graphs(dem: stim.DetectorErrorModel) -> list[list[int]]:
    """
    Return the nodes in the disjoint subgraphs that the DEM (or decoding
    graph) can be split into.

    Notes
    -----
    This function requires ``networkx``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    import networkx as nx

    # convert stim.DetectorErrorModel to nx.Graph
    # to use the functionality 'nx.connected_components(G)'
    g = nx.Graph()
    for dem_instr in dem.flattened():
        if dem_instr.type != "error":
            continue

        targets = dem_instr.targets_copy()
        targets = [t.val for t in targets if t.is_relative_detector_id()]

        if len(targets) == 1:
            g.add_node(targets[0])

        # hyperedges cannot be added to nx.Graph, but if we are just checking
        # the number of disjoint graphs, we can transform the hyperedge to a
        # sequence of edges which keeps the same connectiviy between nodes.
        # For example, hyperedge (0,2,5,6) can be expressed as edges (0,2),
        # (2,5) and (5,6).
        for start, end in zip(targets[:-1], targets[1:]):
            g.add_edge(start, end)

    subgraphs = [list(c) for c in nx.connected_components(g)]

    return subgraphs


def get_flippable_detectors(dem: stim.DetectorErrorModel) -> set[int]:
    """Returns a the detector indices present in the given DEM
    that are triggered by some errors.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    dets = set()
    for dem_instr in dem.flattened():
        if dem_instr.type == "error":
            dets.update(get_detectors(dem_instr))

    return dets


def get_flippable_observables(dem: stim.DetectorErrorModel) -> set[int]:
    """Returns a the logical observable indices present in the given DEM
    that are triggered by some errors.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    obs = set()
    for dem_instr in dem.flattened():
        if dem_instr.type == "error":
            obs.update(get_observables(dem_instr))

    return obs


def contains_only_edges(dem: stim.DetectorErrorModel) -> bool:
    """Returns if the given DEM contains conly edges or boundary edges."""
    for dem_instr in dem.flattened():
        if (dem_instr.type == "error") and len(get_detectors(dem_instr)) > 2:
            return False
    return True


def convert_observables_to_detectors(
    dem: stim.DetectorErrorModel,
    obs_inds: Sequence[int] | None = None,
    det_inds: Sequence[int] | None = None,
) -> stim.DetectorErrorModel:
    """Converts the specified observables into a detector in the specified DEM.

    Parameters
    ----------
    dem
        Detector error model.
    obs_inds
        List of indices of the observables to convert into detectors.
        By default ``None``, which converts all the observables to detectors.
    det_inds
        List of detector indices to which the ``obs_inds`` will be converted.
        By default ``None``, which sets
        ``det_inds = [dem.num_detectors + i for i, _ in enumerate(obs_inds)]``.

    Returns
    -------
    new_dem
        Detector error model with ``obs_inds`` converted to ``det_inds``.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if obs_inds is None:
        obs_inds = list(range(dem.num_observables))
    if not isinstance(obs_inds, list):
        raise TypeError(f"'obs_inds' must be an list, but {type(obs_inds)} was given.")
    if any(not isinstance(o, int) for o in obs_inds):
        raise TypeError("Each element in 'obs_inds' must be an integer.")
    if det_inds is None:
        det_inds = [dem.num_detectors + i for i, _ in enumerate(obs_inds)]
    if not isinstance(det_inds, list):
        raise TypeError(f"'det_inds' must be an list, but {type(det_inds)} was given.")
    if any(not isinstance(d, int) for d in det_inds):
        raise TypeError("Each element in 'det_inds' must be an integer.")

    new_dem = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type == "error":
            detectors = list(get_detectors(instr))
            observables = list(get_observables(instr))

            for obs_ind, det_ind in zip(obs_inds, det_inds):
                if obs_ind in observables:
                    detectors.append(det_ind)
                    observables.remove(obs_ind)

            new_detectors = [stim.target_relative_detector_id(d) for d in detectors]
            new_observables = [
                stim.target_logical_observable_id(l) for l in observables
            ]
            targets = new_detectors + new_observables

            new_instr = stim.DemInstruction(
                type="error",
                args=instr.args_copy(),
                targets=targets,
            )
            new_dem.append(new_instr)
        elif instr.type == "logical_observable":
            obs_ind = instr.targets_copy()[0].val
            if obs_ind not in obs_inds:
                new_dem.append(instr)

            det_ind = det_inds[obs_inds.index(obs_ind)]
            new_instr = stim.DemInstruction(
                type="detector",
                args=[],
                targets=[stim.target_relative_detector_id(det_ind)],
            )
            new_dem.append(new_instr)
        elif instr.type == "detector":
            new_dem.append(instr)
        else:
            raise ValueError(f"Instruction type '{instr.type}' unknown.")

    return new_dem


def get_errors_triggering_detectors(
    dem: stim.DetectorErrorModel, detectors: None | Sequence[int] = None
) -> dict[int, list[int]]:
    """Returns a dictionary that lists all errors that flip each
    specified detector.

    Parameters
    ----------
    dem
        Detector error model.
    detectors
        Detectors for which to compute which errors flip them.
        By default ``None``, which corresponds to all the detectors in ``dem``.

    Returns
    -------
    support
        Dictionary with keys corresponding to ``detectors`` and values corresponding
        the errors that flip the given detector. The errors are repesented as
        indices, corresponding to ``dem.flattened()[i]``.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if detectors is None:
        detectors = list(range(dem.num_detectors))
    if not isinstance(detectors, Sequence):
        raise TypeError(
            f"'detectors' must be a sequence, but {type(detectors)} was given."
        )

    support = {d: [] for d in detectors}
    for error_id, instr in enumerate(dem.flattened()):
        if instr.type != "error":
            continue

        dets = get_detectors(instr)
        for det in dets:
            if det in detectors:
                support[det].append(error_id)

    return support


def only_errors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Returns the corresponding dem with only error instructions."""
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    new_dem = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type == "error":
            new_dem.append(instr)
    return new_dem


def remove_hyperedges(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Removes the hyperedges from the given DEM."""
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    new_dem = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type != "error":
            new_dem.append(instr)
            continue

        if len(get_detectors(instr)) <= 2:
            new_dem.append(instr)

    return new_dem
