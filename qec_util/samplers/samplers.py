from collections.abc import Callable
import time
import pathlib

import numpy as np
import stim

# the package "fcntl" is only available for Unix systems.
FILE_LOCKING = False
try:
    import fcntl

    FILE_LOCKING = True
except:
    pass


def sample_failures(
    dem: stim.DetectorErrorModel,
    decoder,
    min_failures: int | float = 0,
    max_failures: int | float = np.inf,
    min_time: int | float = 0,
    max_time: int | float = np.inf,
    min_samples: int | float = 0,
    max_samples: int | float = np.inf,
    batch_size: int | np.float64 | float | None = None,
    max_batch_size: int | float = np.inf,
    file_name: str | pathlib.Path | None = None,
    decoding_failure: Callable = lambda x: x.any(axis=1),
    extra_metrics: Callable = lambda _: list(),
) -> tuple[int, int, list[int]]:
    """Samples decoding failures until all the minimum requirements have been
    fulfilled (i.e. min. number of failures, min. runtime, min. number of samples)
    and one of three conditions is met:
    (1) max. number of failures reached,
    (2) max. runtime reached,
    (3) max. number of samples taken.

    Parameters
    ----------
    dem
        Detector error model from which to sample the detectors and
        logical observable flips.
    decoder
        Decoder object with a ``decode_batch`` method.
    min_failures
        Minimum number of failures to reach before being able to stop the sampling.
    max_failures
        Maximum number of failures to reach before stopping the calculation.
        By default ``np.inf`` to not have any restriction on the
        maximum number of failures.
    min_time
        Minimum duration for this function (in seconds) before being able to stop the sampling.
    max_time
        Maximum duration for this function, in seconds.
        By default``np.inf`` to not place any restriction on runtime.
    min_failures
        Minimum number of samples to reach before being able to stop the sampling.
    max_samples
        Maximum number of samples to reach before stopping the calculation.
        By default ``np.inf`` to not have any restriction on the
        maximum number of samples.
    batch_size
        Number of samples to decode per batch. If ``None``, it estimates
        the best ``batch_size`` given the other parameters (i.e. ``max_time``,
        ``max_samples``, and ``max_failures``).
    max_batch_size
        Maximum number of samples to decode per batch. This is useful when
        encountering memory issues, as one can just reduce ``max_batch_size``.
    file_name
        Name of the file in which to store the partial results.
        If the file does not exist, it will be created.
        Specifying a file is useful if the computation is stop midway, so
        that it can be continued in if the same file is given. It can also
        be used to sample more points.
    decoding_failure
        Function that returns `True` if there has been a decoding failure, else
        `False`. Its input is an ``np.ndarray`` of shape
        ``(num_samples, num_observables)`` and its output must be a boolean
        ``np.ndarray`` of shape ``(num_samples,)``.
        By default, a decoding failure is when a logical error happened to
        any of the logical observables.
    extra_metrics
        Function that returns a tuple of extra metrics to compute appart
        from the failures. Its input is an ``np.ndarray`` of shape
        ``(num_samples, num_observables)`` and its output must be a boolean
        ``np.ndarray`` of shape ``(num_samples,)``.

    Returns
    -------
    num_failures
        Number of decoding failures.
    num_samples
        Number of samples taken.
    extra_metrics
        Tuple of the extra metrics.

    Notes
    -----
    If ``file_name`` is specified, each batch is stored in the file in a
    different line using the following format: ``num_failures num_samples\n``.
    If extra matrics are calculated, they appear as:
    ``num_failures num_samples | num_extra_metric_1 num_extra_metric_2 ...\n``.
    The number of failures and samples can be read using
    ``read_failures_from_file`` function present in the same module.

    The function will use file locking via ``fcntl`` (only available in Unix systems)
    to avoid having multiple python instances writing on the same file at the same
    time. For other OS, it will run without file locking.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if "decode_batch" not in dir(decoder):
        raise TypeError("'decoder' does not have a 'decode_batch' method.")
    if max_samples == max_time == max_failures == np.inf:
        raise ValueError(
            "One of 'max_samples', 'max_time', or 'max_failures' must be non infinite."
        )

    num_failures, num_samples = 0, 0

    if (file_name is not None) and pathlib.Path(file_name).exists():
        num_failures, num_samples, extra = read_failures_from_file(file_name)
        # check if desired samples/failures have been reached
        if (num_samples >= max_samples) or (num_failures >= max_failures):
            return num_failures, num_samples, extra

    # check that everyting works correct and estimate the batch size for decoding,
    # if needed
    sampler = dem.compile_sampler()
    defects, log_flips, _ = sampler.sample(shots=100)
    t_init = time.time()
    predictions = decoder.decode_batch(defects)
    run_time = (time.time() - t_init) / 100
    log_errors = predictions != log_flips
    failures = decoding_failure(log_errors)
    extra = extra_metrics(log_errors)
    if (not isinstance(failures, np.ndarray)) or (failures.shape != (100,)):
        raise ValueError(
            "'decoding_function' does not return a correctly shaped output."
        )
    if (
        (not isinstance(extra, list))
        or any(not isinstance(m, np.ndarray) for m in extra)
        or any(m.shape != (100,) for m in extra)
    ):
        raise ValueError("'extra_metrics' does not return a correctly shaped output.")

    if batch_size is None:
        log_err_prob = np.average(failures)
        estimated_max_samples = min(
            [
                max_samples - num_samples,
                max_time / run_time,
                (
                    (max_failures - num_failures) / log_err_prob
                    if log_err_prob != 0
                    else np.inf
                ),
            ]
        )
        batch_size = estimated_max_samples / 5  # perform approx 5 batches

        # avoid batch_size = 0 or np.inf and also avoid overshooting
        batch_size = max([batch_size, 1])
        batch_size = min([batch_size, max_samples - num_samples])
        # int(np.inf) raises an error and it could be that both batch_size and
        # max_samples are np.inf
        batch_size = batch_size if batch_size != np.inf else 200_000
        batch_size = min([batch_size, max_batch_size])

    # ensure batch size is int
    batch_size = int(batch_size)

    # initialize the correct size of extra metrics
    extra = [0 for _ in extra]

    # start sampling...
    while (
        (time.time() - t_init) < min_time
        or num_failures < min_failures
        or num_samples < min_samples
        or (
            (time.time() - t_init) < max_time
            and num_failures < max_failures
            and num_samples < max_samples
        )
    ):
        defects, log_flips, _ = sampler.sample(shots=batch_size)
        predictions = decoder.decode_batch(defects)
        log_errors = predictions != log_flips
        batch_failures = decoding_failure(log_errors).sum()
        batch_extra = [int(m.sum()) for m in extra_metrics(log_errors)]

        num_failures += batch_failures
        num_samples += batch_size
        extra = [m + bm for m, bm in zip(extra, batch_extra)]

        if file_name is not None:
            file = open(file_name, "a")
            if FILE_LOCKING:
                fcntl.lockf(file, fcntl.LOCK_EX)

            extra_str = ""
            if len(extra) != 0:
                extra_str = " | " + " ".join([f"{m}" for m in batch_extra])

            file.write(f"{batch_failures} {batch_size}{extra_str}\n")
            file.close()
            # read again num_samples and num_failures to avoid oversampling
            # when multiple processes are writing in the same file.
            num_failures, num_samples, extra = read_failures_from_file(file_name)

    return int(num_failures), num_samples, extra


def read_failures_from_file(
    file_name: str | pathlib.Path,
    max_num_failures: int | float = np.inf,
    max_num_samples: int | float = np.inf,
) -> tuple[int, int, list[int]]:
    """Returns the number of failues and samples stored in a file.

    Parameters
    ----------
    file_name
        Name of the file with the data.
        The structure of the file is specified in the Notes and the intended
        usage is for the ``sample_failures`` function.
    max_num_failues
        If specified, only adds up the first batches until the number of
        failures reaches or (firstly) surpasses the given number.
        By default ``np.inf``, thus it adds up all the batches in the file.
    max_num_samples
        If specified, only adds up the first batches until the number of
        samples reaches or (firstly) surpasses the given number.
        By default ``np.inf``, thus it adds up all the batches in the file.

    Returns
    -------
    num_failures
        Total number of failues in the given number of samples.
    num_samples
        Total number of samples.

    Notes
    -----
    The structure of ``file_name`` file is: each batch is stored in the file in a
    different line using the format ``num_failures num_samples\n``.
    The file ends with an empty line.
    """
    if not pathlib.Path(file_name).exists():
        raise FileExistsError(f"The given file ({file_name}) does not exist.")

    num_failures, num_samples = 0, 0
    extra_metrics = []
    with open(file_name, "r") as file:
        for line in file:
            if line == "":
                continue

            line = line[:-1]  # remove \n character at the end
            if "|" in line:
                # there are extra metrics
                line, extra = line.split(" | ")
                batch_extra_metrics = list(map(int, extra.split(" ")))
                # initialize correct size of extra_metrics
                if len(extra_metrics) == 0:
                    extra_metrics = [0 for _ in batch_extra_metrics]
                extra_metrics = [
                    m + bm for m, bm in zip(extra_metrics, batch_extra_metrics)
                ]

            batch_failures, batch_samples = map(int, line.split(" "))
            num_failures += batch_failures
            num_samples += batch_samples

            if num_failures >= max_num_failures or num_samples >= max_num_samples:
                return num_failures, num_samples, extra_metrics

    return num_failures, num_samples, extra_metrics


def merge_batches_in_file(file_name: str | pathlib.Path) -> None:
    """Merges all the batches in the given file into a single batch,
    which reduces the size of the file.

    Parameters
    ----------
    file_name
        Name of the file with the data.
        The structure of the file is specified in the Notes and the intended
        usage is for the ``sample_failures`` function.
    """
    num_failures, num_samples, extra_metrics = read_failures_from_file(
        file_name=file_name
    )

    with open(file_name, "w") as file:
        extra_str = ""
        if len(extra_metrics) != 0:
            extra_str = " | " + " ".join([f"{m}" for m in extra_metrics])

        file.write(f"{num_failures} {num_samples}{extra_str}\n")

    return
