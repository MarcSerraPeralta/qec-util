from collections.abc import Iterable

import stim

from .util import xor_lists


def get_detectors(dem_instr: stim.DemInstruction) -> tuple[int, ...]:
    """Returns the detector indices that are flipped in the given DEM instruction."""
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise ValueError(f"DemInstruction is not an error, it is {dem_instr.type}.")

    if has_separator(dem_instr):
        return xor_lists(*decomposed_detectors(dem_instr))
    else:
        return tuple(
            sorted(
                i.val for i in dem_instr.targets_copy() if i.is_relative_detector_id()
            )
        )


def get_logicals(dem_instr: stim.DemInstruction) -> tuple[int, ...]:
    """Returns the logical observable indices that are flipped in the given DEM instruction."""
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise ValueError(f"DemInstruction is not an error, it is {dem_instr.type}.")

    if has_separator(dem_instr):
        return xor_lists(*decomposed_logicals(dem_instr))
    else:
        return tuple(
            sorted(
                i.val for i in dem_instr.targets_copy() if i.is_logical_observable_id()
            )
        )


def has_separator(dem_instr: stim.DemInstruction) -> bool:
    """Returns if the given DEM instruction has a separator."""
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise ValueError(f"DemInstruction is not an error, it is {dem_instr.type}.")

    return bool([i for i in dem_instr.targets_copy() if i.is_separator()])


def decomposed_detectors(dem_instr: stim.DemInstruction) -> list[tuple[int, ...]]:
    """Returns a list of the detector indices triggered for each fault that the DEM
    instruction is decomposed into.
    """
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise ValueError(f"DemInstruction is not an error, it is {dem_instr.type}.")

    list_dets = []
    current = []
    for e in dem_instr.targets_copy():
        if e.is_separator():
            list_dets.append(current)
            current = []
        if e.is_relative_detector_id():
            current.append(e.val)
    list_dets.append(current)

    # process dets
    list_dets = [tuple(sorted(d)) for d in list_dets]

    return list_dets


def decomposed_logicals(dem_instr: stim.DemInstruction) -> list[tuple[int, ...]]:
    """Returns a list of the logical indices triggered for each fault that the DEM
    instruction is decomposed into.
    """
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        raise ValueError(f"'dem_instr' is not an error, it is {dem_instr.type}.")

    list_logs = []
    current = []
    for e in dem_instr.targets_copy():
        if e.is_separator():
            list_logs.append(current)
            current = []
        if e.is_logical_observable_id():
            current.append(e.val)
    list_logs.append(current)

    # process dets
    list_logs = [tuple(sorted(l)) for l in list_logs]

    return list_logs


def remove_detectors(
    dem_instr: stim.DemInstruction, dets: Iterable[int]
) -> stim.DemInstruction:
    """Removes the specified detector indices from the given DEM instruction."""
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if not isinstance(dets, Iterable):
        raise TypeError(f"'dets' must be iterable, but {type(dets)} was given.")

    if dem_instr.type != "error":
        raise ValueError(f"'dem_instr' is not an error, it is {dem_instr.type}.")

    prob = dem_instr.args_copy()
    targets = [
        d
        for d in dem_instr.targets_copy()
        if not (d.is_relative_detector_id() and (d.val in dets))
    ]

    # recurrently check that there cannot be any separator at the beginning or
    # end of a stim.DemInstruction.
    correct = [False, False]
    while not (correct[0] and correct[1]):
        if (len(targets) != 0) and targets[0].is_separator():
            targets = targets[1:]
        else:
            correct[0] = True

        if (len(targets) != 0) and targets[-1].is_separator():
            targets = targets[:-1]
        else:
            correct[1] = True

    return stim.DemInstruction(type="error", targets=targets, args=prob)


def sorted_dem_instr(dem_instr: stim.DemInstruction) -> stim.DemInstruction:
    """Returns the dem_instr in an specific order.
    Note that it removes the separators.
    """
    if not isinstance(dem_instr, stim.DemInstruction):
        raise TypeError(
            f"'dem_instr' must be a stim.DemInstruction, but {type(dem_instr)} was given."
        )
    if dem_instr.type != "error":
        return dem_instr

    dets = sorted(get_detectors(dem_instr))
    logs = sorted(get_logicals(dem_instr))
    dets_target = list(map(stim.target_relative_detector_id, dets))
    logs_target = list(map(stim.target_logical_observable_id, logs))
    prob = dem_instr.args_copy()

    return stim.DemInstruction(
        type="error", targets=dets_target + logs_target, args=prob
    )
