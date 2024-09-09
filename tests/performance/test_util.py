import numpy as np
from uncertainties import UFloat

from qec_util.performance import util


def test_util():
    methods = [
        "logical_error_prob",
        "logical_error_prob_decay",
        "LogicalErrorProbDecayModel",
        "lmfit_par_to_ufloat",
        "confidence_interval_binomial",
    ]

    assert set(dir(util)) >= set(methods)


def test_logical_error_prob():
    predictions = np.array([0, 1, 0, 1])
    true_values = np.array([0, 0, 0, 1])

    log_err_prob = util.logical_error_prob(predictions, true_values)
    assert log_err_prob == 0.25

    return


def test_logical_error_prob_decay():
    log_err_prob = util.logical_error_prob_decay(1, 0.1, qec_offset=1)
    assert log_err_prob == 0
    return


def test_logicalfidelitydecay_fit():
    qec_round = np.arange(10)
    error_rate = 0.01
    qec_offset = 2
    log_err_prob = util.logical_error_prob_decay(
        qec_round, error_rate, qec_offset=qec_offset
    )

    model = util.LogicalErrorProbDecayModel(vary_qec_offset=True)
    params = model.guess(log_err_prob, qec_round)
    result = model.fit(
        log_err_prob, params=params, qec_round=qec_round, min_qec_round=2
    )
    fitted_error_rate = result.params["error_rate"].value
    fitted_qec_offset = result.params["qec_offset"].value

    assert np.isclose(fitted_error_rate, error_rate)
    assert np.isclose(fitted_qec_offset, qec_offset)

    return


def test_lmfit_par_to_ufloat():
    qec_round = np.arange(10)
    error_rate = 0.01
    qec_offset = 2
    log_err_prob = util.logical_error_prob_decay(
        qec_round, error_rate, qec_offset=qec_offset
    )

    model = util.LogicalErrorProbDecayModel(vary_qec_offset=True)
    params = model.guess(log_err_prob, qec_round)
    result = model.fit(log_err_prob, params=params, qec_round=qec_round)
    fitted_error_rate = result.params["error_rate"]

    error_rate_ufloat = util.lmfit_par_to_ufloat(fitted_error_rate)
    assert isinstance(error_rate_ufloat, UFloat)
    assert np.isclose(error_rate_ufloat.nominal_value, error_rate)
    assert error_rate_ufloat.std_dev is not None

    return


def test_confidence_interval_binomial():
    num_failures, num_samples = 1, 100
    average = num_failures / num_samples
    lower_bound, upper_bound = util.confidence_interval_binomial(
        num_failures, num_samples, probit=1
    )
    assert lower_bound > 0 and lower_bound < average
    assert upper_bound > 0 and upper_bound > average
    return
