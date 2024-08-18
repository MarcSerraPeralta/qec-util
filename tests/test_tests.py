import os
import pathlib


DIR_EXCEPTIONS = ["__pycache__"]
FILE_EXCEPTIONS = ["__init__.py"]


def test_tests():
    test_dir = pathlib.Path("tests")
    mod_dir = pathlib.Path("qec_util")
    if not mod_dir.exists():
        raise ValueError("module directory does not exist.")

    for path, _, files in os.walk(mod_dir):
        for file in files:
            if file[-3:] != ".py" or file[0] == "_":
                continue

            # change root dir from "iq_readout" to test_dir
            relpath = os.path.relpath(path, mod_dir)
            testpath = os.path.join(test_dir, relpath)
            if file not in FILE_EXCEPTIONS:
                if not os.path.exists(os.path.join(testpath, "test_" + file)):
                    raise ValueError(
                        f"test file for {os.path.join(mod_dir, file)}" " does not exist"
                    )
    return
