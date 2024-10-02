import stim


def remove_gauge_detectors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
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
