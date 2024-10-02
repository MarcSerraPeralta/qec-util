import stim


def remove_gauge_detectors(circuit: stim.Circuit) -> stim.Circuit:
    """Removes the gauge detectors from the given circuit."""
    dem = circuit.detector_error_model(allow_gauge_detectors=True)
    gauge_dets = []
    for dem_instr in dem.flattened():
        if dem_instr.type == "error" and dem_instr.args_copy() == [0.5]:
            if len(dem_instr.targets_copy()) != 1:
                raise ValueError("There exist 'composed' gauge detector: {dem_instr}.")
            gauge_dets.append(dem_instr.targets_copy()[0].val)

    if len(gauge_dets) == 0:
        return circuit

    current_det = -1
    new_circuit = stim.Circuit()
    for instr in circuit.flattened():
        if instr.name == "DETECTOR":
            current_det += 1
            if current_det in gauge_dets:
                continue

        new_circuit.append(instr)

    return new_circuit
