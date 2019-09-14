# Diagnostic efficiency

Info: de needs Python >= 3.6!

##### Development branch
[![Build Status](https://travis-ci.com/schwemro/de.svg?token=xpMVcD4f5rphE6dVCxpb&branch=master)](https://travis-ci.com/schwemro/de)
[![codecov](https://codecov.io/gh/schwemro/de/branch/master/graph/badge.svg)](https://codecov.io/gh/schwemro/de)

## How to cite

In case you use SciKit-GStat in other software or scientific publications,
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

sig_de = de.calc_de(obs_arr, sim_arr)

de.vis2d_de(obs_arr, sim_arr)
```
