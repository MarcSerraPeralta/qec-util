import pytest


def pytest_addoption(parser):
    parser.addoption("--show-figures", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    # The hyphens in the arguments are substutited by underscores.
    option_value = metafunc.config.option.show_figures
    if "show_figures" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("show_figures", [bool(option_value)])


@pytest.fixture(scope="session")
def failures_file(tmp_path_factory):
    """This function is executed before any test is run and creates
    a file in a temporary file that can be passed to any test function.

    From https://docs.pytest.org/en/stable/how-to/tmp_path.html
    """
    file_name = tmp_path_factory.mktemp("data") / "sample_failures_file.txt"
    contents = "10 50\n11 50\n"
    with open(file_name, "w") as file:
        file.write(contents)
    return file_name
