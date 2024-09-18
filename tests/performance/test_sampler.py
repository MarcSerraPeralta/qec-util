import time

import numpy as np
import stim
from pymatching import Matching

from qec_util.performance import sample_failures, read_failures_from_file


def test_sampler_to_file(tmp_path):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples = sample_failures(
        dem,
        mwpm,
        max_samples=1_000,
        max_time=np.inf,
        max_failures=np.inf,
        file_name=tmp_path / "tmp_file.txt",
    )
    read_failures, read_samples = read_failures_from_file(tmp_path / "tmp_file.txt")

    assert num_failures == read_failures
    assert num_samples == read_samples

    return


def test_sample_early_stopping(failures_file):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples = sample_failures(
        dem,
        mwpm,
        max_samples=1,
        max_time=np.inf,
        max_failures=1,
        file_name=failures_file,
    )

    assert num_failures == 21
    assert num_samples == 100

    return


def test_sampler_from_file(failures_file):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples = sample_failures(
        dem,
        mwpm,
        max_samples=100,
        max_time=np.inf,
        max_failures=np.inf,
        file_name=failures_file,
    )

    assert num_failures == 21
    assert num_samples == 100

    return


def test_sampler():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples = sample_failures(
        dem, mwpm, max_samples=1_000, max_time=np.inf, max_failures=np.inf
    )
    assert num_samples >= 1_000
    assert (num_failures >= 0) and (num_samples) >= 0

    t_init = time.time()
    num_failures, num_samples = sample_failures(
        dem, mwpm, max_samples=np.inf, max_time=1.1, max_failures=np.inf
    )
    assert time.time() - t_init >= 1.1
    assert (num_failures >= 0) and (num_samples) >= 0

    num_failures, num_samples = sample_failures(
        dem, mwpm, max_samples=np.inf, max_time=np.inf, max_failures=10
    )
    assert num_failures >= 10
    assert (num_failures >= 0) and (num_samples) >= 0

    return


def test_read_failures_from_file(failures_file):
    num_failures, num_samples = read_failures_from_file(failures_file)
    assert num_failures == 21
    assert num_samples == 100
    return
