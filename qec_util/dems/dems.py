import stim


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
    num_dets = dem_1.num_detectors
    for dem_instr in dem_1.flattened():
        if dem_instr.type != "error":
            continue

        # remove separators
        targets = [t for t in dem_instr.targets_copy() if not t.is_separator()]

        # dem_instr1 == dem_instr2 is only true if the argument is the same
        # and the targets are sorted also in the same way. For example
        # "error(0.1) D0 D1" is different than "error(0.1) D1 D0".
        def order_1(t) -> int:
            if t.is_logical_observable_id():
                return num_dets + t.val
            if t.is_relative_detector_id():
                return t.val
            else:
                raise NotImplemented(f"{t} is not a logical or a detector.")

        targets = sorted(targets, key=order_1)
        prob = dem_instr.args_copy()[0]
        dem_1_ordered.append("error", prob, targets)

    dem_2_ordered = stim.DetectorErrorModel()
    num_dets = dem_2.num_detectors
    for dem_instr in dem_2.flattened():
        if dem_instr.type != "error":
            continue

        # remove separators
        targets = [t for t in dem_instr.targets_copy() if not t.is_separator()]

        # dem_instr1 == dem_instr2 is only true if the argument is the same
        # and the targets are sorted also in the same way. For example
        # "error(0.1) D0 D1" is different than "error(0.1) D1 D0".
        def order_2(t) -> int:
            if t.is_logical_observable_id():
                return num_dets + t.val
            if t.is_relative_detector_id():
                return t.val
            else:
                raise NotImplemented(f"{t} is not a logical or a detector.")

        targets = sorted(targets, key=order_2)
        prob = dem_instr.args_copy()[0]
        dem_2_ordered.append("error", prob, targets)

    diff_1 = stim.DetectorErrorModel()
    for dem_instr in dem_1_ordered:
        if dem_instr not in dem_2_ordered:
            diff_1.append(dem_instr)

    diff_2 = stim.DetectorErrorModel()
    for dem_instr in dem_2_ordered:
        if dem_instr not in dem_1_ordered:
            diff_2.append(dem_instr)

    return diff_1, diff_2
