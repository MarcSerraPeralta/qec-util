def sorting_index(t, num_dets) -> int:
    """Function to sort the logical and detector targets inside a DEM instruction.

    Parameters
    ----------
    t
        stim.DemTarget.
    num_dets
        Number of detectors in the DEM.

    Returns
    -------
    Sorting index associated with ``t``.

    Notes
    -----
    ``dem_instr1 == dem_instr2`` is only true if the argument is the same
    and the targets are sorted also in the same way. For example
    ``"error(0.1) D0 D1"`` is different than ``"error(0.1) D1 D0"``.
    """
    if t.is_logical_observable_id():
        return num_dets + t.val
    if t.is_relative_detector_id():
        return t.val
    else:
        raise NotImplemented(f"{t} is not a logical or a detector.")
