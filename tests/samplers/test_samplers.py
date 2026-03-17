import pathlib
import time

import numpy as np
import pytest
import stim
from pymatching import Matching

from qec_util.samplers import (
    merge_batches_in_file,
    merge_files,
    read_failures_from_file,
    sample_failures,
)


def test_sampler_to_file(tmp_path: pathlib.Path):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples_ps, num_samples, extra = sample_failures(
        dem,
        mwpm,
        max_samples=1_000,
        max_time=np.inf,
        max_failures=np.inf,
        file_name=tmp_path / "tmp_file.csv",
    )
    read_failures, read_samples_ps, read_samples, read_extra = read_failures_from_file(
        tmp_path / "tmp_file.csv"
    )

    assert num_failures == read_failures
    assert num_samples_ps == read_samples_ps
    assert num_samples == read_samples
    assert extra == read_extra
    assert "seconds" in extra

    with open(tmp_path / "tmp_file.csv") as file:
        header = file.readline()[:-1]

    assert header == "num_failures_ps,num_samples_ps,num_samples,seconds"

    return


def test_sampler_extra_metrics(tmp_path: pathlib.Path):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)
    extra_metrics = lambda x: {"test": np.zeros(len(x), dtype=bool)}

    num_failures, num_samples_ps, num_samples, extra = sample_failures(
        dem,
        mwpm,
        max_samples=100_000,
        file_name=tmp_path / "tmp_file_extra_metrics.csv",
        verbose=False,
        extra_metrics=extra_metrics,
    )

    read_failures, read_samples_ps, read_samples, read_extra = read_failures_from_file(
        tmp_path / "tmp_file_extra_metrics.csv"
    )

    assert num_failures == read_failures
    assert num_samples_ps == read_samples_ps
    assert num_samples == read_samples
    assert extra == read_extra
    assert extra["test"] == 0
    assert extra["seconds"] > 0

    num_failures, num_samples_ps, num_samples, extra = sample_failures(
        dem,
        mwpm,
        max_samples=100_000,
        file_name=tmp_path / "tmp_file_extra_metrics.csv",
        verbose=False,
        extra_metrics_ps=extra_metrics,
    )

    read_failures, read_samples_ps, read_samples, read_extra = read_failures_from_file(
        tmp_path / "tmp_file_extra_metrics.csv"
    )

    assert num_failures == read_failures
    assert num_samples_ps == read_samples_ps
    assert num_samples == read_samples
    assert extra == read_extra
    assert extra["test"] == 0
    assert extra["seconds"] > 0

    with open(tmp_path / "tmp_file_extra_metrics.csv") as file:
        header = file.readline()[:-1]

    assert header == "num_failures_ps,num_samples_ps,num_samples,seconds,test"

    with pytest.raises(ValueError):
        _ = sample_failures(
            dem,
            mwpm,
            max_samples=100_000,
            file_name=tmp_path / "tmp_file_extra_metrics.csv",
            verbose=False,
            extra_metrics=extra_metrics,
            extra_metrics_ps=extra_metrics,
        )

    return


def test_samples_minimum_requirements():
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    _, _, num_samples, _ = sample_failures(
        dem,
        mwpm,
        max_samples=200,
        min_samples=200,
        max_time=0,
        max_failures=1,
        batch_size=10,
    )
    assert num_samples == 200

    num_failures, _, _, _ = sample_failures(
        dem,
        mwpm,
        min_failures=20,
        max_failures=20,
        max_samples=1,
        max_time=0,
    )
    assert num_failures >= 20

    return


def test_sample_early_stopping(failures_file: str):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem,
        mwpm,
        max_samples=1,
        max_time=np.inf,
        max_failures=1,
        file_name=failures_file,
    )

    assert num_failures == 21
    assert num_samples_ps == 100
    assert num_samples == 100

    return


def test_sampler_from_file(failures_file: str):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)

    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem,
        mwpm,
        max_samples=100,
        max_time=np.inf,
        max_failures=np.inf,
        file_name=failures_file,
    )

    assert num_failures == 21
    assert num_samples_ps == 100
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

    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem, mwpm, max_samples=1_000, max_time=np.inf, max_failures=np.inf
    )
    assert num_samples_ps >= 1_000
    assert num_samples >= 1_000
    assert (num_failures >= 0) and (num_samples) >= 0

    t_init = time.time()
    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem, mwpm, max_samples=np.inf, max_time=0.1, max_failures=np.inf
    )
    assert time.time() - t_init >= 0.1
    assert (num_failures >= 0) and (num_samples) >= 0 and (num_samples_ps >= 0)

    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem,
        mwpm,
        max_failures=10,
        batch_size=100_000,
    )
    assert num_failures >= 10
    assert (num_failures >= 0) and (num_samples >= 0) and (num_samples_ps >= 0)

    num_failures, num_samples_ps, num_samples, _ = sample_failures(
        dem,
        mwpm,
        max_samples=2_000,
        batch_size=1_000,
        post_selection=lambda x: np.zeros(len(x), dtype=bool),
    )
    assert num_samples_ps == 0
    assert num_samples == 2_000

    return


def test_read_failures_from_file(failures_file: str):
    num_failures, num_samples_ps, num_samples, _ = read_failures_from_file(
        failures_file
    )
    assert num_failures == 21
    assert num_samples_ps == 100
    assert num_samples == 100
    return


def test_merge_batches_in_file(tmp_path: pathlib.Path):
    circuit = stim.Circuit.generated(
        code_task="repetition_code:memory",
        distance=3,
        rounds=3,
        after_clifford_depolarization=0.01,
    )
    dem = circuit.detector_error_model()
    mwpm = Matching(dem)
    extra_metrics = lambda x: {"test": np.zeros(len(x), dtype=bool)}

    num_failures, num_samples_ps, num_samples, extra = sample_failures(
        dem,
        mwpm,
        max_samples=10,
        batch_size=1,
        max_time=np.inf,
        max_failures=np.inf,
        file_name=tmp_path / "tmp_file_merge_batches.csv",
        extra_metrics=extra_metrics,
    )
    merge_batches_in_file(tmp_path / "tmp_file_merge_batches.csv")

    with open(tmp_path / "tmp_file_merge_batches.csv", "r") as file:
        num_lines = len([l for l in file.read().split("\n") if l != ""])

    assert num_lines == 2  # header + merged line

    read_failures, read_samples_ps, read_samples, read_extra = read_failures_from_file(
        tmp_path / "tmp_file_merge_batches.csv"
    )

    assert num_failures == read_failures
    assert num_samples_ps == read_samples_ps
    assert num_samples == read_samples
    assert extra["test"] == read_extra["test"]
    assert np.isclose(extra["seconds"], read_extra["seconds"])

    return


def test_merge_files(tmp_path: pathlib.Path):
    contents = "num_failures_ps,num_samples_ps,num_samples,seconds,m1,m2\n"
    contents += "1,10,10,0.003,2,4\n"
    with open(tmp_path / "tmp_file_1.csv", "w") as file:
        file.write(contents)

    contents = "num_failures_ps,num_samples_ps,num_samples,seconds,m1,m2\n"
    contents += "5,20,20,0.004,1,0\n"
    with open(tmp_path / "tmp_file_2.csv", "w") as file:
        file.write(contents)

    merge_files(
        [tmp_path / "tmp_file_1.csv", tmp_path / "tmp_file_2.csv"],
        tmp_path / "merged_file.csv",
    )

    num_failures, num_samples_ps, num_samples, extra = read_failures_from_file(
        tmp_path / "merged_file.csv"
    )
    assert num_failures == 6
    assert num_samples_ps == 30
    assert num_samples == 30
    assert extra["m1"] == 3
    assert extra["m2"] == 4
    assert len(extra) == 3

    assert (tmp_path / "tmp_file_1.csv").exists()
    assert (tmp_path / "tmp_file_2.csv").exists()

    merge_files(
        [tmp_path / "tmp_file_1.csv", tmp_path / "tmp_file_2.csv"],
        tmp_path / "merged_file.csv",
        delete_files=True,
    )

    assert not (tmp_path / "tmp_file_1.csv").exists()
    assert not (tmp_path / "tmp_file_2.csv").exists()

    contents = "num_failures_ps,num_samples_ps,num_samples,seconds,m1,m2\n"
    contents += "5,20,20,0.004,1,0\n"
    with open(tmp_path / "merged_file.csv", "w") as file:
        file.write(contents)

    merge_files(
        [tmp_path / "merged_file.csv", tmp_path / "merged_file.csv"],
        tmp_path / "merged_file.csv",
    )

    num_failures, num_samples_ps, num_samples, extra = read_failures_from_file(
        tmp_path / "merged_file.csv"
    )
    assert num_failures == 5
    assert num_samples_ps == 20
    assert num_samples == 20
    assert extra["m1"] == 1
    assert extra["m2"] == 0
    assert len(extra) == 3

    return
