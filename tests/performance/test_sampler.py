import time

import stim
from pymatching import Matching

from qec_util.performance import sample_failures


def test_sampler():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.001,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    t_init = time.time()
    num_failures, num_samples = sample_failures(
        dem, mwpm, max_samples=1_000, max_time=10, max_failures=100
    )
    assert (
        (num_samples >= 1_000)
        or ((time.time() - t_init) >= 10)
        or (num_failures >= 100)
    )

    return
