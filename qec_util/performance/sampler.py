from typing import Tuple
import time

import numpy as np
import stim


def sample_failures(
    dem: stim.DetectorErrorModel,
    decoder,
    max_failures: int = 100,
    max_time: int = 3600,
    max_samples: int = 1_000_000,
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
    max_time
        Maximum duration for this function, in seconds. Set this parameter
        to ``np.inf`` to not place any restriction on runtime.
    max_samples
        Maximum number of samples to reach before stopping the calculation.

    Returns
    -------
    num_failures
        Number of decoding failures.
    num_samples
        Number of samples taken.
    """
    if not isinstance(dem, stim.DetectorErrorModel):
        raise TypeError(
            f"'dem' must be a stim.DetectorErrorModel, but {type(dem)} was given."
        )
    if "decode_batch" not in dir(decoder):
        raise TypeError("'decoder' does not have a 'decode_batch' method.")

    sampler = dem.compile_sampler()
    num_failures, num_samples = 0, 0

    # estimate the batch size for decoding
    defects, log_flips, _ = sampler.sample(shots=100)
    t_init = time.time()
    predictions = decoder.decode_batch(defects)
    run_time = (time.time() - t_init) / 100
    log_err_prob = np.average(predictions != log_flips)
    estimated_max_samples = min(
        [
            max_samples,
            max_time / run_time,
            max_failures / log_err_prob if log_err_prob != 0 else np.inf,
        ]
    )
    batch_size = int(estimated_max_samples / 10)

    # start sampling...
    while (
        (time.time() - t_init) < max_time
        and num_failures < max_failures
        and num_samples < max_samples
    ):
        defects, log_flips, _ = sampler.sample(shots=batch_size)
        predictions = decoder.decode_batch(defects)
        log_errors = predictions != log_flips
        num_failures += log_errors.sum()
        num_samples += batch_size

    return int(num_failures), num_samples
