import stim

from ..dem_instrs import (
    decompose_hyperedge_to_edges,
    decomposed_instrs,
    get_detectors,
    has_separator,
    sorted_dem_instr,
    xor_probs,
)
from .dems import only_errors, remove_hyperedges


def decompose_hyperedges_to_edges(
    dem: stim.DetectorErrorModel,
    dem_edges: stim.DetectorErrorModel | None = None,
    ignore_decomposition_failures: bool = False,
) -> stim.DetectorErrorModel:
    """Decomposes the hyperedges from the given detector model into edges using
    Algorithm 3 from https://doi.org/10.48550/arXiv.2309.15354.

    Paramteres
    ----------
    dem
        Detector error model with hyperedges to decompose.
    dem_edges
        Errors to use for the decomposition of the hyperedges.
        If this DEM contains hyperedges, they will be ignored.
        By default ``None``, which uses corresponds to all the edges
        and boundary edges of ``dem``.
    ignore_decomposition_failures
        If ``True``, does not raises an error if any hyperedge decomposition does not
        match the logical observable effect of the hyperedge.
        By default ``False``.

    Returns
    -------
    decomposed_dem
        Detector error model with a suggestion for the decomposition of
        the hyperedges described with ``stim.target_separator``s.

    Notes
    -----
    If hyperedges contains decompositions with ``stim.target_separator``s,
    they are going to be overwritten.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if dem_edges is None:
        dem_edges = only_errors(dem)
        dem_edges = remove_hyperedges(dem_edges)
    if not isinstance(dem_edges, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem_edges' must be a stim.DetectorErrorModel, but {type(dem_edges)} was given."
        )

    decomposed_dem = stim.DetectorErrorModel()
    for instr in dem:
        if instr.type != "error" or len(get_detectors(instr)) <= 2:
            decomposed_dem.append(instr)
        else:
            decomposed_dem.append(
                decompose_hyperedge_to_edges(
                    instr,
                    dem_edges,
                    ignore_decomposition_failure=ignore_decomposition_failures,
                )
            )

    return decomposed_dem


def decomposed_graphlike_dem(
    dem: stim.DetectorErrorModel, prob_method: str = "same"
) -> stim.DetectorErrorModel:
    """Returns the decomposed DEM (containing only edges) for the given DEM.

    Parameters
    ----------
    dem
        Detector error model.
    prob_method
        Method for assigning probabilities to the decomposed errors.
        See ``qec_util.dem_instrs.decomposed_instrs`` for more information.
        By default, the same probability of the hyperedge is used for each
        of its associated edges.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )

    edges = {}
    hyperedges = []
    attributes_dem = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type != "error":
            attributes_dem.append(instr)
            continue

        # it could be possible that an edge has been decomposed...
        if has_separator(instr):
            hyperedges.append(instr)
            continue

        if len(get_detectors(instr)) <= 2:
            sorted_instr = sorted_dem_instr(instr, prob=0)
            prob = instr.args_copy()[0]
            if sorted_instr not in edges:
                edges[sorted_instr] = 0
            edges[sorted_instr] = xor_probs(prob, edges[sorted_instr])
        else:
            hyperedges.append(instr)

    for hyperedge in hyperedges:
        for edge in decomposed_instrs(hyperedge, prob_method=prob_method):
            if len(get_detectors(edge)) > 2:
                raise ValueError(f"Non-decomposed hyperedge found in dem: {edge}.")

            sorted_instr = sorted_dem_instr(edge, prob=0)
            if sorted_instr not in edges:
                raise ValueError(
                    f"Edge '{edge}' found in a decomposition is not an existing edge in the DEM."
                )

            prob = edge.args_copy()[0]
            edges[sorted_instr] = xor_probs(prob, edges[sorted_instr])

    decomposed_dem = stim.DetectorErrorModel()
    for edge, prob in edges.items():
        instr = stim.DemInstruction("error", args=[prob], targets=edge.targets_copy())
        decomposed_dem.append(instr)

    decomposed_dem += attributes_dem

    return decomposed_dem
