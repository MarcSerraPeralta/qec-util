from typing import Tuple, Optional, Callable
import time
import pathlib

import numpy as np
import stim


def sample_failures(
    dem: stim.DetectorErrorModel,
    decoder,
    max_failures: int | float = 100,
    max_time: int | float = np.inf,
    max_samples: int | float = 1_000_000,
    file_name: Optional[str | pathlib.Path] = None,
    decoding_failure: Callable = lambda x: x.any(axis=1),
) -> Tuple[int, int]:
    """Samples decoding failures until one of three conditions is met:
    (1) max. number of failures reached, (2) max. runtime reached,
    (3) max. number of samples taken.

    Parameters
    ----------
    dem
        Detector error model from which to sample the detectors and
        logical flips.
    decoder
        Decoder object with a ``decode_batch`` method.
    max_failures
        Maximum number of failures to reach before stopping the calculation.
        Set this parameter to ``np.inf`` to not have any restriction on the
        maximum number of failures.
    max_time
        Maximum duration for this function, in seconds. By default, this
        parameter is set to ``np.inf`` to not place any restriction on runtime.
    max_samples
        Maximum number of samples to reach before stopping the calculation.
        Set this parameter to ``np.inf`` to not have any restriction on the
        maximum number of samples.
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

    Returns
    -------
    num_failures
        Number of decoding failures.
    num_samples
        Number of samples taken.

    Notes
    -----
    If ``file_name`` is specified, each batch is stored in the file in a
    different line using the following format: ``num_failures num_samples\n``.
    The number of failures and samples can be read using
    ``read_failures_from_file`` function present in the same module.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if "decode_batch" not in dir(decoder):
        raise TypeError("'decoder' does not have a 'decode_batch' method.")

    num_failures, num_samples = 0, 0

    if (file_name is not None) and pathlib.Path(file_name).exists():
        num_failures, num_samples = read_failures_from_file(file_name)
        # check if desired samples/failures have been reached
        if (num_samples >= max_samples) or (num_failures >= max_failures):
            return num_failures, num_samples

    # estimate the batch size for decoding
    sampler = dem.compile_sampler()
    defects, log_flips, _ = sampler.sample(shots=100)
    t_init = time.time()
    predictions = decoder.decode_batch(defects)
    run_time = (time.time() - t_init) / 100
    failures = decoding_failure(predictions != log_flips)
    if (not isinstance(failures, np.ndarray)) or (failures.shape != (100,)):
        raise ValueError(
            f"'decoding_function' does not return a correctly shaped output"
        )
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
    batch_size = int(batch_size)

    # start sampling...
    while (
        (time.time() - t_init) < max_time
        and num_failures < max_failures
        and num_samples < max_samples
    ):
        defects, log_flips, _ = sampler.sample(shots=batch_size)
        predictions = decoder.decode_batch(defects)
        log_errors = predictions != log_flips
        batch_failures = decoding_failure(log_errors).sum()

        num_failures += batch_failures
        num_samples += batch_size

        if file_name is not None:
            with open(file_name, "a") as file:
                file.write(f"{batch_failures} {batch_size}\n")
            # read again num_samples and num_failures to avoid oversampling
            # when multiple processes are writing in the same file.
            num_failures, num_samples = read_failures_from_file(file_name)

    return int(num_failures), num_samples


def read_failures_from_file(
    file_name: str | pathlib.Path,
    max_num_failures: int | float = np.inf,
    max_num_samples: int | float = np.inf,
) -> Tuple[int, int]:
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
    with open(file_name, "r") as file:
        for line in file:
            if line == "":
                continue

            line = line[:-1]  # remove \n character at the end
            batch_failures, batch_samples = map(int, line.split(" "))
            num_failures += batch_failures
            num_samples += batch_samples

            if num_failures >= max_num_failures or num_samples >= max_num_samples:
                return num_failures, num_samples

    return num_failures, num_samples
