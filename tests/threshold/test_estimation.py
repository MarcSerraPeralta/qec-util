from qec_util.threshold import get_threshold


def test_get_threshold():
    p = [0, 0.1, 0.2, 0.3]
    data = {
        3: (p, [0, 10, 20, 30], [1000, 234, 234, 235]),
        5: (p, [0, 4, 25, 45], [2000, 234, 234, 240]),
        7: (p, [0, 1, 36, 60], [1234, 234, 234, 241]),
    }

    p_thr, ci_lower, ci_upper = get_threshold(data)

    assert 0.1 <= p_thr <= 0.2
    assert ci_lower < p_thr < ci_upper

    p_thr, ci_lower, ci_upper = get_threshold(data, num_samples_bootstrap=1)

    assert 0.1 <= p_thr <= 0.2
    assert ci_lower < p_thr < ci_upper

    data = {
        3: (p, [0, 10, 20, 30], [1000, 234, 234, 235]),
        5: (p, [0, 4, 22, 45], [2000, 234, 234, 240]),
        7: ([0, 0.1, 0.2, 0.3, 0.4], [0, 1, 25, 60, 80], [1234, 234, 234, 241, 250]),
    }

    p_thr, ci_lower, ci_upper = get_threshold(data, num_samples_bootstrap=0)

    assert 0.1 <= p_thr <= 0.2
    assert ci_lower < p_thr < ci_upper

    return
