from qec_util.data_processing import syndrome


def test_syndrome():
    methods = ["get_syndromes", "get_defects", "get_final_defects"]

    assert set(dir(syndrome)) >= set(methods)

    return
