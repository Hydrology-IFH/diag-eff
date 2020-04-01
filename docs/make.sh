#!/bin/bash

# set path to project folder
cd /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg/docs

sphinx-build -b html /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg/docs /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg/docs/_build

# make docs
make html
# make latexpdf
# make man
