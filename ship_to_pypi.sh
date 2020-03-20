#!/bin/bash

# set path to project folder
cd /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg

# git tags are required for versioning
# git tag 0.0.1
# git push origin master --tags

# clean up repository for packaging
# git -xfd

# generating distribution archives
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# uploading the distribution archives
python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository-url https://pypi.org/legacy/ dist/*

# installing your newly uploaded package
pip3 install de

# further steps:
# python
# import de