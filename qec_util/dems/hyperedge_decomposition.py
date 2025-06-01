import stim

from .dems import remove_hyperedges, only_errors
from ..dem_instrs import get_detectors, decompose_hyperedge_to_edges


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
