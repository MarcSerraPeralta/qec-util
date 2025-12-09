from qec_util.data_processing import syndrome


def test_syndrome():
    methods = [
        "get_syndromes",
        "get_defects",
        "get_final_defects",
        "get_defect_probs",
        "get_final_defect_probs",
    ]

    assert set(dir(syndrome)) >= set(methods)

    return
