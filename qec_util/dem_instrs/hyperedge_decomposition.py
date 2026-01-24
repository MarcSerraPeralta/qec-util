import numpy as np
import stim

from .dem_instrs import get_detectors, get_observables


def decompose_hyperedge_to_edges(
    hyperedge: stim.DemInstruction,
    dem_edges: stim.DetectorErrorModel,
    ignore_decomposition_failure: bool = False,
) -> stim.DemInstruction:
    """Decomposes a single hyperedge using the given edges and using
    Algorithm 3 from https://doi.org/10.48550/arXiv.2309.15354.

    Parameters
    ----------
    hyperedge
        The hyperedge to decompose into edges.
    dem_edges
        Edge errors to use for the decomposition of the hyperedges.
    ignore_decomposition_failures
        If ``True``, does not raises an error if any hyperedge decomposition does not
        match the logical observable effect of the hyperedge.
        By default ``False``.

    Notes
    -----
    This function requires ``pymatching``. To install the requirements to be able
    to execute any function in ``qec_util``, run ``pip install qec_util[all]``.

    If the hyperedge contains a decomposition with ``stim.target_separator``s,
    it is going to be overwritten.
    """
    if not isinstance(hyperedge, stim.DemInstruction):
        raise TypeError(
            f"'hyperedge' must be a stim.DemInstruction, but {type(hyperedge)} was given."
        )
    if hyperedge.type != "error":
        raise TypeError(
            f"'hyperedge' must be an error, but {hyperedge.type} was given."
        )
    if not isinstance(dem_edges, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem_edges' must be a stim.DetectorErrorModel, but {type(dem_edges)} was given."
        )
    dem_edges = dem_edges.flattened()
    for instr in dem_edges:
        if instr.type != "error" or len(get_detectors(instr)) > 2:
            raise TypeError(
                f"'dem_edges' must only contain edge errors, but {instr} was found. "
                "Use 'remove_hyperedges(only_edges(dem))' to only have edge errors."
            )
    detectors = get_detectors(hyperedge)
    if max(detectors) > dem_edges.num_errors:
        raise ValueError(
            f"'dem_edges' do not span the whole detectors required for 'hyperedge'."
        )

    from pymatching import Matching

    mwpm = Matching(dem_edges)
    syndrome = np.zeros(dem_edges.num_detectors, dtype=bool)
    syndrome[np.array(detectors)] = True
    edges = mwpm.decode_to_edges_array(syndrome)

    # convert edges array to a list of stim.DemInstructions
    edge_to_instr = {}
    for instr in dem_edges:
        dets = tuple(sorted(get_detectors(instr)))
        dets = dets if len(dets) == 2 else (-1, dets[0])
        edge_to_instr[dets] = instr

    decomposition = stim.DetectorErrorModel()
    for edge in edges:
        edge = tuple(sorted(edge))
        decomposition.append(edge_to_instr[edge])

    if not ignore_decomposition_failure:
        obs = set()
        for error in decomposition:
            obs.symmetric_difference_update(get_observables(error))
        if obs != set(get_observables(hyperedge)):
            raise ValueError(
                f"Decomposition with different logical observable effect found for {hyperedge}:"
                f"\n{decomposition}"
            )

    # build decomposed hyperedge
    targets = []
    for error in decomposition:
        targets += error.targets_copy()
        targets.append(stim.target_separator())
    targets = targets[:-1]  # remove last separator
    decom_hyperedge = stim.DemInstruction(
        "error", targets=targets, args=hyperedge.args_copy()
    )

    return decom_hyperedge
