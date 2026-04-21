import math
from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def xor_two_lists(list1: Iterable[T], list2: Iterable[T]) -> tuple[T, ...]:
    """Returns the symmetric difference of two lists.
    Note that the resulting list has been sorted.
    """
    return tuple(sorted(set(list1).symmetric_difference(list2)))


def xor_lists(*elements: Iterable[T]) -> tuple[T]:
    """Returns the symmetric difference of multiple lists.
    Note that the resulting list has been sorted.
    """
    output = []
    for element in elements:
        output = xor_two_lists(output, element)
    return tuple(sorted(output))


def xor_two_probs(p: float | int, q: float | int) -> float | int:
    """Returns the probability of only one of the events happening.

    Parameters
    ----------
    p
        Probability of one event.
    q
        Probability of the other event.
    """
    return p * (1 - q) + (1 - p) * q


def xor_probs(*probs: float | int) -> float | int:
    """Returns the probability of an odd number of events happening.

    Parameters
    ----------
    *probs
        Probabilities of each of the events.
    """
    odd_prob = probs[0]
    for prob in probs[1:]:
        odd_prob = xor_two_probs(prob, odd_prob)
    return odd_prob


def prob_indep_depol1(p):
    """Returns the independent probability for an individual mechanism (e.g. ``Y``)
    in a ``DEPOLARIZE1`` channel. This is the probability that Stim uses to
    decompose the ``DEPOLARIZE1`` channel into ``X_ERROR``, ``Y_ERROR``, and
    ``Z_ERROR``. See https://quantumcomputing.stackexchange.com/questions/45779
    for more information.

    Parameters
    ----------
    p
        Probability (argument) of the ``DEPOLARIZE1`` channel.
    """
    return 0.5 - 0.5 * math.sqrt(1 - 4 * p / 3)


def prob_indep_depol2(p):
    """Returns the independent probability for an individual mechanism (e.g. ``Y``)
    in a ``DEPOLARIZE2`` channel. This is the probability that Stim uses to
    decompose the ``DEPOLARIZE2`` channel. See
    https://quantumcomputing.stackexchange.com/questions/45779 for more information.

    Parameters
    ----------
    p
        Probability (argument) of the ``DEPOLARIZE2`` channel.
    """
    return 0.5 - 0.5 * (1 - 16 * p / 15) ** 0.125
