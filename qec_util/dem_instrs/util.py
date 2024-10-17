from collections.abc import Iterable


def xor_two_lists(list1: Iterable, list2: Iterable) -> tuple:
    """Returns the symmetric difference of two lists.
    Note that the resulting list has been sorted.
    """
    return tuple(sorted(set(list1).symmetric_difference(list2)))


def xor_lists(*elements: Iterable) -> tuple:
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
