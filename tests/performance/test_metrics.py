import numpy as np
from uncertainties import UFloat

from qec_util.performance import metrics


def test_metrics():
    methods = [
        "logical_error_prob",
        "logical_error_prob_decay",
        "LogicalErrorProbDecayModel",
        "lmfit_par_to_ufloat",
        "confidence_interval_binomial",
    ]

    assert set(dir(metrics)) >= set(methods)


def test_logical_error_prob():
    predictions = np.array([0, 1, 0, 1])
    true_values = np.array([0, 0, 0, 1])

    log_err_prob = metrics.logical_error_prob(predictions, true_values)
    assert log_err_prob == 0.25

    return


def test_logical_error_prob_decay():
    log_err_prob = metrics.logical_error_prob_decay(1, 0.1, round_offset=1)
    assert log_err_prob == 0
    return


def test_get_error_rate():
    rounds = np.arange(10)
    error_rate = 0.01
    round_offset = 2
    log_err_prob = metrics.logical_error_prob_decay(
        rounds, error_rate, round_offset=round_offset
    )

    fitted_error_rate, fitted_round_offset = metrics.get_error_rate(
        rounds, log_err_prob, return_round_offset=True
    )

    assert np.isclose(fitted_error_rate.nominal_value, error_rate)
    assert np.isclose(fitted_round_offset.nominal_value, round_offset)

    return


def test_logicalfidelitydecay_fit():
    rounds = np.arange(10)
    error_rate = 0.01
    round_offset = 2
    log_err_prob = metrics.logical_error_prob_decay(
        rounds, error_rate, round_offset=round_offset
    )

    model = metrics.LogicalErrorProbDecayModel(vary_round_offset=True)
    params = model.guess(log_err_prob, rounds)
    result = model.fit(log_err_prob, params=params, rounds=rounds, min_round_fit=2)
    fitted_error_rate = result.params["error_rate"].value
    fitted_round_offset = result.params["round_offset"].value

    assert np.isclose(fitted_error_rate, error_rate)
    assert np.isclose(fitted_round_offset, round_offset)

    return


def test_lmfit_par_to_ufloat():
    rounds = np.arange(10)
    error_rate = 0.01
    round_offset = 2
    log_err_prob = metrics.logical_error_prob_decay(
        rounds, error_rate, round_offset=round_offset
    )

    model = metrics.LogicalErrorProbDecayModel(vary_round_offset=True)
    params = model.guess(log_err_prob, rounds)
    result = model.fit(log_err_prob, params=params, rounds=rounds)
    fitted_error_rate = result.params["error_rate"]

    error_rate_ufloat = metrics.lmfit_par_to_ufloat(fitted_error_rate)
    assert isinstance(error_rate_ufloat, UFloat)
    assert np.isclose(error_rate_ufloat.nominal_value, error_rate)
    assert error_rate_ufloat.std_dev is not None

    return


def test_confidence_interval_binomial():
    num_failures, num_samples = 1, 100
    average = num_failures / num_samples
    lower_bound, upper_bound = metrics.confidence_interval_binomial(
        num_failures, num_samples, probit=1
    )
    assert lower_bound > 0 and lower_bound < average
    assert upper_bound > 0 and upper_bound > average
    return
