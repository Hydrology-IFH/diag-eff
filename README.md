# Diagnostic efficiency <img src="logo.png" align="right" width="120" />

Info: `de` needs Python >= 3.6!

##### Development branch
[![Build Status](https://travis-ci.com/schwemro/de.svg?token=xpMVcD4f5rphE6dVCxpb&branch=master)](https://travis-ci.com/schwemro/de)
[![Documentation Status](https://readthedocs.org/projects/de/badge/?version=latest)](https://de.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/schwemro/de/branch/master/graph/badge.svg?token=AmLX6d2FuR)](https://codecov.io/gh/schwemro/de)
[![PyPI version shields.io](https://img.shields.io/pypi/v/de.svg)](https://pypi.python.org/pypi/de/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Binder](http://mybinder.org/badge_logo.svg)](http://mybinder.org/v2/gh/binder-examples/conda_environment/master?filepath=index.ipynb)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)
[![DOI:10.](https://zenodo.org/badge/DOI/.svg)](https://doi.org/)

## How to cite

In case you use de in other software or scientific publications,
please reference this module. It is published and has a DOI. It can be cited
as:
    Schwemmle, R., Demand, D., and Weiler, M.: Diagnostic efficiency – specific
    evaluation of model performance, Hydrol. Earth Syst. Sci., X, xxxx-xxxx,
    DOI: [xxx](https://doi.org/xxxx), 2020.

## Full Documentation

The full documentation can be found at: https://de.readthedocs.io

## License
This software can be distributed freely under the GPL v3 license. Please read the LICENSE for further information.

© 2019, Robin Schwemmle (<robin.schwemmle@hydrology.uni-freiburg.de>)

## Description

`de` is an open-source toolbox written in Python for specific evaluation of
model performance. The toolbox provides functions to calculate the Diagnostic
Efficiency metric and and fucntions to visualize contribution of metric terms
by diagnostic polar plots. Additionally, functions to calculate KGE and NSE
are available.

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
from pathlib import Path  # OS-independent path handling
from de import de
from de import util

# set path to example data
path_cam = Path('/examples/camels_example_data/13331500_94_model_output.txt')

# import example data as dataframe
df_cam = util.import_camels_obs_sim(path_cam)

# make arrays
obs_arr = df_cam['Qobs'].values
sim_arr = df_cam['Qsim'].values

# calculate diagnostic efficiency
sig_de = de.calc_de(obs_arr, sim_arr)

# diagnostic polar plot
de.diag_polar_plot(obs_arr, sim_arr)
```
## Usage in R

In order to run `de` in R, **reticulate** can be used as an interface to Python.

Non-interactive mode:

```r
install.packages("reticulate")
library(reticulate)

# pip installation
py_install("numpy")
py_install("pandas")
py_install("scipy")
py_install("matplotlib")
py_install("seaborn")
py_install("de")

# import Python modules
os <- import("os")
np <- import("numpy")
pd <- import("pandas")
sp <- import("scipy")
mpl <- import("matplotlib")
plt <- import("matplotlib.pyplot")
sns <- import("seaborn")
de <- import("de")

# set path to example data
path_cam <- file.path(path_wd,
                    'examples/camels_example_data/13331500_94_model_output.txt')

# import example data as dataframe
df_cam <- import_camels_obs_sim(path_cam)

# calculate diagnostic efficiency
sig_de <- calc_de(df_cam$Qobs, df_cam$Qsim)

# diagnostic polar plot
fig <- diag_polar_plot(df_cam$Qobs, df_cam$Qsim)
# currently figures cannot be displayed interactively in a R environment
fig$savefig('diagnostic_polar_plot.png')
```

Interactive mode using a Python interpreter in R:

```r
install.packages("reticulate")
library(reticulate)

# pip installation
py_install("numpy")
py_install("pandas")
py_install("scipy")
py_install("matplotlib")
py_install("seaborn")
py_install("tk")
py_install("de")

# start Python interpreter in R
repl_python()
```
```python
# copy+paste the lines below to the interpreter
import os
PATH = '/Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg'
os.chdir(PATH)
from de import de
from de import util
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# set path to example data
path = os.path.join(os.getcwd(), 'examples/camels_example_data/13331500_94_model_output.txt')

# import example data as dataframe
df_cam = util.import_camels_obs_sim(path)

# make arrays
obs_arr = df_cam['Qobs'].values
sim_arr = df_cam['Qsim'].values

plt.plot(obs_arr)
plt.show()

# calculate diagnostic efficiency
sig_de = de.calc_de(obs_arr, sim_arr)

# diagnostic polar plot
de.diag_polar_plot(obs_arr, sim_arr)
plt.show()

# quit the interpreter
exit
```
