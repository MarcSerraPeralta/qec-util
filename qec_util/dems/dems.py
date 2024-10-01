import stim


def remove_gauge_detectors(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    new_dem = stim.DetectorErrorModel()

    for dem_instr in dem.flattened():
        if dem_instr.type != "error":
            new_dem.append(dem_instr)

        if dem_instr.args_copy() != [0.5]:
            new_dem.append(dem_instr)

    return new_dem
