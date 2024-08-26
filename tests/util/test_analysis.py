from qec_util.util import analysis


def test_analysis():
    methods = [
        "error_prob",
        "error_prob_decay",
        "logical_fidelity",
        "logical_fidelity_decay",
        "LogicalFidelityDecay",
        "lmfit_par_to_ufloat",
    ]

    assert set(dir(analysis)) >= set(methods)
