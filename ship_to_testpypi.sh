#!/bin/bash

# set path to project folder
cd /Users/robinschwemmle/Desktop/PhD/diagnostic_efficiency/pkg

# git tags are required for versioning
# git tag 0.0.1
# git push origin master --tags

# clean up repository for packaging
# create branch for version
# git checkout -b 0.0.1
# git -xfd

# generating distribution archives
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# uploading the distribution archives
python3 -m pip install --user --upgrade twine
python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose

# installing your newly uploaded package
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps de-schwemro

# further steps:
# python
# import de