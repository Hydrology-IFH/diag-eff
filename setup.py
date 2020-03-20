"""Setup for the de package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import versioneer

long_description = """de is an open-source toolbox written in Python for
specific evaluation of model performance. The toolbox provides functions to
calculate the Diagnostic Efficiency metric and and fucntions to visualize
contribution of metric terms by diagnostic polar plots. Note that the data
management of time series is handled using pandas data frame objects.
"""

INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "matplotlib",
    "seaborn",
    "pandas",
]
TEST_REQUIRES = [
    # testing and coverage
    "pytest",
    "coverage",
    "pytest-cov",
    # to be able to run `python setup.py checkdocs`
    "collective.checkdocs",
    "pygments",
]

setup(
    name="de",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="DE: Diagnostic Efficiency",
    long_description=long_description,
    url="https://github.com/schwemro/de",
    author="Robin Schwemmle, Dominic Demand, Markus Weiler",
    author_email="robin.schwemmle@hydrology.uni-freiburg.de",
    license="GPLv3",
    classifiers=[
        "Development Status :: 4 - Beta ",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology ",
        "Topic :: Scientific/Engineering :: Visualization",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3) ",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    python_requires=">=3.6",
    packages=find_packages(exclude=["docs"]),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require={"test": TEST_REQUIRES + INSTALL_REQUIRES,},
)
