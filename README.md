# Diagnostic efficiency <img src="logo.png" align="right" width="120" />

Info: de needs Python >= 3.6!

##### Development branch
[![Build Status](https://travis-ci.com/schwemro/de.svg?token=xpMVcD4f5rphE6dVCxpb&branch=master)](https://travis-ci.com/schwemro/de)
[![Documentation Status](https://readthedocs.org/projects/de/badge/?version=latest)](http://ansicolortags.readthedocs.io/?badge=latest)
[![codecov](https://codecov.io/gh/schwemro/de/branch/master/graph/badge.svg)](https://codecov.io/gh/schwemro/de)
[![PyPI version shields.io](https://img.shields.io/pypi/v/de.svg)](https://pypi.python.org/pypi/de/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Binder](http://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/binder-examples/conda_environment/master?filepath=index.ipynb)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![DOI:10.](https://zenodo.org/badge/DOI/.svg)](https://doi.org/)

## How to cite

In case you use de in other software or scientific publications,
please reference this module. It is published and has a DOI. It can be cited
as:
    ...

## Full Documentation

The full documentation can be found at: https://schwemro.github.io/de

## License
This software can be distributed freely under the GPL v3 license. Please read the LICENSE for further information.

© 2019, Robin Schwemmle (<robin.schwemmle@hydrology.uni-freiburg.de>)

## Description

...

## Installation
PyPI:

```bash
pip install de
```


GIT:

```bash
git clone https://github.com/schwemro/de.git
cd de
pip install -e .
```

## Usage

```python
from de import de
from de import util

path = '.../obs_sim.csv'
df_ts = util.import_ts(path, sep=';')

obs_arr = df_ts['Qobs'].values
sim_arr = df_ts['Qsim'].values

# calculate diagnostic efficiency
sig_de = de.calc_de(obs_arr, sim_arr)

# diagnostic polar plot
de.vis2d_de(obs_arr, sim_arr)
```
