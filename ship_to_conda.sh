#!/bin/bash

# set path to project folder
cd /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/diag-eff

# upload to anaconda
anaconda upload dist/*.tar.gz

# package is now available at: https://anaconda.org/schwemro/diag-eff
