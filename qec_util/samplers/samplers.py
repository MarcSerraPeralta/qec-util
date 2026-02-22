import csv
import os
import pathlib
import time
from collections.abc import Callable, Sequence
from datetime import datetime

import numpy as np
import numpy.typing as npt
import stim

# the package "fcntl" is only available for Unix systems.
FILE_LOCKING = False
try:
    import fcntl

    FILE_LOCKING = True
except ImportError:
    pass

BinVect = npt.NDArray[np.bool_]
ExtraMetrics = dict[str, BinVect]

HEADER = ["num_failures_ps", "num_samples_ps", "num_samples", "seconds"]


def sample_failures(
    dem: stim.DetectorErrorModel,
    decoder,
    min_failures: int | float = 0,
    max_failures: int | float = np.inf,
    min_time: int | float = 0,
    max_time: int | float = np.inf,
    min_samples: int | float = 0,
    max_samples: int | float = np.inf,
    min_samples_ps: int | float = 0,
    max_samples_ps: int | float = np.inf,
    batch_size: int = 1_000,
    file_name: str | pathlib.Path | None = None,
    decoding_failure: Callable[[BinVect], BinVect] = lambda x: x.any(axis=1),
    post_selection: Callable[[BinVect], BinVect] = lambda x: np.ones(
        len(x), dtype=bool
    ),
    extra_metrics: Callable[[BinVect], ExtraMetrics] = lambda _: {},
    extra_metrics_ps: Callable[[BinVect], ExtraMetrics] = lambda _: {},
    verbose: bool = True,
) -> tuple[int, int, int, dict[str, int | float]]:
    """Samples decoding failures until ALL the MINIMUM requirements have been
    fulfilled (i.e. ``min_failures``, ``min_time``, ``min_samples``, ``min_samples_ps``)
    and ONE of the MAXIMUM requirements has been fulfilled:
    ``max_failures``, ``max_time``, ``max_samples``, ``max_samples_ps``.

    By default, all the minimum requirements are always fulfilled,
    and the sampling runs until the end of time (unless the job is killed).

    Parameters
    ----------
    dem
        Detector error model from which to sample the detectors and
        logical observable flips.
    decoder
        Decoder object with a ``decode_batch`` method.
    min_failures
        Minimum number of failures (after post-selection, if enabled)
        to reach before being able to stop the sampling.
        By default ``0``, so this requirement is always fulfilled.
    max_failures
        Maximum number of failures (after post-selection, if enabled)
        to reach before stopping the calculation.
        By default ``np.inf`` to not have any restriction on the
        maximum number of failures.
    min_time
        Minimum duration for this function (in seconds) before being able to stop the sampling.
        By default ``0``, so this requirement is always fulfilled.
    max_time
        Maximum duration for this function, in seconds.
        By default``np.inf`` to not place any restriction on runtime.
    min_samples
        Minimum number of samples to reach before being able to stop the sampling.
        By default ``0``, so this requirement is always fulfilled.
    max_samples
        Maximum number of samples to reach before stopping the calculation.
        By default ``np.inf`` to not have any restriction on the
        maximum number of samples.
    min_samples_ps
        Minimum number of post-selected samples to reach before being able to stop the sampling.
        By default ``0``, so this requirement is always fulfilled.
    max_samples_ps
        Maximum number of post-selected samples to reach before stopping the calculation.
        By default ``np.inf`` to not have any restriction on the
        maximum number of samples.
    batch_size
        Number of samples to decode per batch. By default ``1_000``.
    file_name
        Name of the CSV file in which to store the partial results.
        If the file does not exist, it will be created.
        Specifying a file is useful if the computation is stop midway, so
        that it can be continued in if the same file is given.
    decoding_failure
        Function that returns ``True`` if there has been a decoding failure, else
        ``False``. Its input is an ``np.ndarray`` of shape
        ``(num_samples_ps, num_observables)`` and its output must be a boolean
        ``np.ndarray`` of shape ``(num_samples_ps,)``.
        By default, a decoding failure corresponds to a logical error happening to
        any of the logical observables.
    post_selection
        Function that returns ``True`` if the sample needs to be kept, and
        ``False`` if it needs to be discarded. Its input is an ``np.ndarray`` of shape
        ``(num_samples, num_observables)`` and its output must be a boolean
        ``np.ndarray`` of shape ``(num_samples,)``.
        By default, all samples are kept (no post-selection).
    extra_metrics
        Function that returns a dictionary of extra metrics to compute appart
        from the failures before post-selection. Its input is an ``np.ndarray`` of shape
        ``(num_samples, num_observables)`` and its output must be a dictionary
        mapping strings to boolean ``np.ndarray``s of shape ``(num_samples,)``.
        By default, the decoding runtime is stored as ``"seconds"`` (which
        cannot be overwritten).
        The keys in ``extra_metrics`` must be different than ones in
        ``extra_metrics_ps``.
    extra_metrics_ps
        Function that returns a dictionary of extra metrics to compute appart
        from the failures after post-selection. Its input is an ``np.ndarray`` of shape
        ``(num_samples_ps, num_observables)`` and its output must be a dictionary
        mapping strings to boolean ``np.ndarray``s of shape ``(num_samples_ps,)``.
        By default, the decoding runtime is stored as ``"seconds"`` (which
        cannot be overwritten).
        The keys in ``extra_metrics_ps`` must be different than ones in
        ``extra_metrics``.
    verbose
        Flag to print information during sampling. By default, ``False``.

    Returns
    -------
    num_failures
        Number of decoding failures after post-selection.
    num_samles_ps
        Number of samples kept after post-selection.
    num_samples
        Number of samples taken.
    extra_metrics
        Dictionary of the extra metrics.

    Notes
    -----
    If ``file_name`` is specified, each batch is stored in the CSV file in a
    different line. The structure of the CSV file is:

        ``num_failures_ps, num_samples_ps, num_samples, seconds, **extra metrics``
        ``            int,            int,         int,   float,          int(s)``

    The information in the CSV file can be read using the
    ``read_failures_from_file`` function present in this same module.

    The function will use file locking via ``fcntl`` (only available in Unix systems)
    to avoid having multiple python instances writing on the same file at the same
    time. For any other OS, it will run without file locking.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if "decode_batch" not in dir(decoder):
        raise TypeError("'decoder' does not have a 'decode_batch' method.")
    if not isinstance(batch_size, int):
        raise TypeError(
            f"'batch_size' must be an int, but {type(batch_size)} was given."
        )
    vars = [min_failures, min_time, min_samples, min_samples_ps]
    vars += [max_failures, max_time, max_samples, max_samples_ps]
    for var in vars:
        if not isinstance(var, (int, float)):
            raise TypeError(
                "Minimum and maximum requirements must be an int or float, "
                f"but {type(var)} was given."
            )

    num_failures, num_samples_ps, num_samples, runtime = 0, 0, 0, 0

    def print_v(string: str):
        if verbose:
            print(datetime.now(), string)
        return

    def finished() -> bool:
        min_req = (
            (num_failures >= min_failures)
            and (num_samples_ps >= min_samples_ps)
            and (num_samples >= min_samples)
            and (runtime >= min_time)
        )
        max_req = (
            (num_failures >= max_failures)
            or (num_samples_ps >= max_samples_ps)
            or (num_samples >= max_samples)
            or (runtime >= max_time)
        )
        return min_req and max_req

    # check output format of functions
    test = np.zeros((batch_size, dem.num_observables), dtype=bool)
    test_failures = decoding_failure(test)
    test_ps = post_selection(test)
    test_metrics = extra_metrics(test)
    test_metrics_ps = extra_metrics_ps(test)
    for var in (
        test_failures,
        test_ps,
        *test_metrics.values(),
        *test_metrics_ps.values(),
    ):
        if not isinstance(var, np.ndarray):
            raise TypeError(
                "Incorrect output type of the decoding failures, metrics, or post-selection: "
                f"{type(var)}."
            )
        if not isinstance(var.dtype, np.dtypes.BoolDType):
            raise TypeError(
                "Incorrect output type of the decoding failures, metrics, or post-selection: "
                f"{type(var.dtype)}."
            )
        if var.shape != (batch_size,):
            raise TypeError(
                "Incorrect output size of the decoding failures, metrics, or post-selection: "
                f"got {var.shape} expected {(batch_size,)}."
            )

    metric_names = list(test_metrics) + list(test_metrics_ps)
    if "seconds" in metric_names:
        raise ValueError("'seconds' cannot be used a metric name.")
    if len(set(metric_names)) != len(metric_names):
        raise ValueError(
            "'extra_metrics' and 'extra_metrics_ps' must have different keys."
        )
    extra: dict[str, int | float] = {k: 0 for k in metric_names}
    extra["seconds"] = 0.0

    if file_name is not None:
        if pathlib.Path(file_name).exists():
            print_v("File already exists, reading file...")
            num_failures, num_samples_ps, num_samples, extra = read_failures_from_file(
                file_name
            )
            if set(extra) != set(["seconds"] + metric_names):
                raise ValueError(
                    "The metrics in this function and the ones in the file do not match."
                )

            # check if desired samples/failures have been reached
            if finished():
                print_v("File has enough samples and failures.")
                return num_failures, num_samples_ps, num_samples, extra
        else:
            # add header
            print_v("Opening file to store header...")
            _write_header(file_name, metric_names)

    print_v("Compile sampler from DEM...")
    sampler = dem.compile_sampler()

    # start sampling...
    while not finished():
        print_v(f"Sampling {batch_size} shots...")
        defects, log_flips, _ = sampler.sample(shots=batch_size)
        num_samples += batch_size
        print_v(f"Decoding {batch_size} shots...")
        t0 = time.time()
        predictions = decoder.decode_batch(defects)
        t1 = time.time()
        batch_runtime = t1 - t0
        runtime += batch_runtime
        log_errors = predictions != log_flips
        print_v("Post-selecting samples...")
        post_selected = post_selection(log_errors)
        log_errors_ps = log_errors[post_selected]
        batch_samples_ps = int(post_selected.sum())
        num_samples_ps += batch_samples_ps
        print_v(f"There were {batch_samples_ps} shots kept from {batch_size} shots.")
        print_v("Computing decoding failures...")
        batch_failures = int(decoding_failure(log_errors_ps).sum())
        num_failures += batch_failures
        print_v(
            f"There were {batch_failures} failures in {batch_samples_ps} kept shots."
        )
        print_v("Evaluating extra metrics...")
        batch_extra: dict[str, int | float] = {
            k: int(m.sum()) for k, m in extra_metrics(log_errors).items()
        }
        batch_extra |= {
            k: int(m.sum()) for k, m in extra_metrics_ps(log_errors_ps).items()
        }
        batch_extra["seconds"] = batch_runtime
        extra = {k: extra[k] + batch_extra[k] for k in ["seconds"] + metric_names}

        if file_name is not None:
            print_v("Opening file to store data...")
            _append_data(
                file_name, batch_failures, batch_samples_ps, batch_size, batch_extra
            )

            # read again data from file to avoid oversampling
            # when multiple processes are writing in the same file.
            print_v("Update data in case multiple processes are running...")
            num_failures, num_samples_ps, num_samples, extra = read_failures_from_file(
                file_name
            )

    print_v("Sampling conditions are reached, finished sampling.")
    return num_failures, num_samples_ps, num_samples, extra


def _write_header(file_name: str | pathlib.Path, metric_names: Sequence[str]):
    file = open(file_name, "w")
    if FILE_LOCKING:
        fcntl.lockf(file, fcntl.LOCK_EX)

    # sort metrics names to always store them in the same order
    header = ",".join(HEADER + sorted(metric_names)) + "\n"
    _ = file.write(header)
    file.close()
    return


def _append_data(
    file_name: str | pathlib.Path,
    num_failures: int,
    num_samples_ps: int,
    num_samples: int,
    extra: dict[str, int | float],
):
    file = open(file_name, "a")
    if FILE_LOCKING:
        fcntl.lockf(file, fcntl.LOCK_EX)

    _extra = extra.copy()
    runtime = _extra.pop("seconds")

    data = f"{num_failures},{num_samples_ps},{num_samples},{runtime:0.6f}"
    # sort metrics names to always store them in the same order
    for m in sorted(_extra):
        data += f",{_extra[m]}"
    data += "\n"

    _ = file.write(data)
    file.close()
    return


def read_failures_from_file(
    file_name: str | pathlib.Path,
    max_num_failures: int | float = np.inf,
    max_num_samples_ps: int | float = np.inf,
    max_num_samples: int | float = np.inf,
) -> tuple[int, int, int, dict[str, int | float]]:
    """Returns the number of failues and samples stored in a file.

    Parameters
    ----------
    file_name
        Name of the file with the data.
        The structure of the file is specified in the Notes and the intended
        usage is for the ``sample_failures`` function.
    max_num_failues
        If specified, only adds up the first batches until the number of
        failures (after post-selection) reaches or (firstly) surpasses the given number.
        By default ``np.inf``, thus it adds up all the batches in the file.
    max_num_samples_ps
        If specified, only adds up the first batches until the number of
        samples (after post-selection) reaches or (firstly) surpasses the given number.
        By default ``np.inf``, thus it adds up all the batches in the file.
    max_num_samples
        If specified, only adds up the first batches until the number of
        samples (without post-selection) reaches or (firstly) surpasses the given number.
        By default ``np.inf``, thus it adds up all the batches in the file.

    Returns
    -------
    num_failures
        Total number of post-selected failues.
    num_samples_ps
        Number of post-selected samples.
    num_samples
        Total number of samples.
    extra_metrics
        Dictionary of the extra metrics.

    Notes
    -----
    See documentation in ``sample_failures`` function in this module.
    """
    if not pathlib.Path(file_name).exists():
        raise FileExistsError(f"The given file ({file_name}) does not exist.")

    num_failures, num_samples_ps, num_samples = 0, 0, 0
    with open(file_name, "r") as file:
        reader = csv.reader(file, delimiter=",")
        header = next(reader, None)
        if header is None:
            raise ValueError("Header is missing in CSV file")
        if (len(header) < len(HEADER)) or (header[: len(HEADER)] != HEADER):
            raise ValueError("Incorrect header.")
        extra_metrics: dict[str, int | float] = {k: 0 for k in header[3:]}

        for row in reader:
            num_failures += int(row[0])
            num_samples_ps += int(row[1])
            num_samples += int(row[2])
            extra_metrics["seconds"] += float(row[3])
            for k, v in zip(header[len(HEADER) :], row[len(HEADER) :]):
                extra_metrics[k] += int(v)

            if num_failures >= max_num_failures:
                break
            if num_samples_ps >= max_num_samples_ps:
                break
            if num_samples >= max_num_samples:
                break

    return num_failures, num_samples_ps, num_samples, extra_metrics


def merge_batches_in_file(file_name: str | pathlib.Path) -> None:
    """Merges all the batches in the given file into a single batch,
    which reduces the size of the file.

    Parameters
    ----------
    file_name
        Name of the file with the data.
        The structure of the file is specified in the Notes from
        ``read_failures_from_file`` function and the intended usage is for the
        ``sample_failures`` function.
    """
    num_failures, num_samples_ps, num_samples, extra_metrics = read_failures_from_file(
        file_name=file_name
    )
    _write_header(file_name, list(extra_metrics))
    _append_data(file_name, num_failures, num_samples_ps, num_samples, extra_metrics)
    return


def merge_files(
    file_names: list[str | pathlib.Path],
    merged_file_name: str | pathlib.Path,
    delete_files: bool = False,
) -> None:
    """Merges the batches in the given files into a single file.
    Batches in each file are aggregated into a single line in the new file.

    Parameters
    ----------
    file_names
        Name of the files with the data.
        The structure of the file is specified in the Notes from
        ``read_failures_from_file`` function and the intended usage is for the
        ``sample_failures`` function.
    merged_file_name
        Name of the file to merge the files into.
    delete_files
        Flag to delete the files after being merged. By default ``False``.
    """
    # do not merge the merged_file_name data into merged_file_name
    # as this would correspond to duplicating the data.
    if merged_file_name in file_names:
        file_names = [f for f in file_names if f != merged_file_name]

    # remove duplicate copies as data would be duplicated.
    for file_name in set(file_names):
        num_failures, num_samples_ps, num_samples, extra_metrics = (
            read_failures_from_file(file_name=file_name)
        )

        if not pathlib.Path(merged_file_name).exists():
            metric_names = [n for n in extra_metrics if n != "seconds"]
            _write_header(merged_file_name, metric_names)

        _append_data(
            merged_file_name, num_failures, num_samples_ps, num_samples, extra_metrics
        )

        if delete_files:
            os.remove(file_name)

    return
