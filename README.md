# qec-util

![example workflow](https://github.com/MarcSerraPeralta/qec-util/actions/workflows/actions.yaml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![PyPI](https://img.shields.io/pypi/v/qec_util?label=pypi%20package)

A collection of utility methods and objects to aid with the simulation, decoding and analysis of QEC experiments.


## Installation

This package is available in PyPI, thus it can be installed using
```
pip install qec-util
```
or alternatively, it can be installed from source using
```
git clone git@github.com:MarcSerraPeralta/qec-util.git
pip install qec-util/
```

## Setting up the gurobi license

1. Create a free academic account
2. Request a license, which will give you a license key
3. Install the Gurobi Optimizer (or install `gurobipy` through conda) so that we can run the `grbgetkey` command
4. Run `grbgetkey xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` using your license number
5. At the end of `~/.bashrc` add `export GRB_LICENSE_FILE=/path/to/license.lic` where the license path is printed when running the previous step
6. Run `source ~/.bashrc` or open a new terminal and check that the license installation is successful by running the `gurobi.sh` command
