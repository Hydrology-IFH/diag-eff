#!/bin/bash

# set path to project folder
cd /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/diag-eff

# git tags are required for versioning
# git tag 0.1
# git push origin master --tags

# clean up repository for packaging. add files before ignoring.
# create branch for version
# git checkout -b 0.1
# git clean -xfd

# generating distribution archives
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# uploading the distribution archives
python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository-url https://pypi.org/legacy/ dist/*
# after revisions adjust versioneer style in setup.cfg

# installing your newly uploaded package
pip3 install diag-eff

# further steps:
# python
# import de
