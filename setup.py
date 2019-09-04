"""Setup for the de package."""

# !/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

long_description = """DE is an open-source toolbox written in Python to diagnose
model performance. The toolbox provides functions to calculate the Diagnostic
Efficiency measure and visualize the three components on which the measure
is based in 2D-space. First, we introduce a novel model performance metric.
Secondly, visualising the three metric components in a 2D-space origin of
errors either input data or model structure/parameterization can be easily
distinguished. Moreover, we provide functions to manipulate the observed
hydrologic time series. These manipulations either mimick model errors or input
data errors. Hence, deliver a proof of concept. Note that the data
management of time series is handled using pandas data frame objects.
"""

INSTALL_REQUIRES = [
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'pandas',
]
TEST_REQUIRES = [
    # testing and coverage
    'pytest', 'coverage', 'pytest-cov',
    # to be able to run `python setup.py checkdocs`
    'collective.checkdocs', 'pygments',
]

setup(
    name='de',

    version='0.1',

    description='DE: Diagnosing model Efficiency',
    long_description=long_description,

    url='https://github.com/schwemro/de',

    author='Robin Schwemmle, Markus Weiler, Dominic Demand, Andreas Hartmann',
    author_email='robin.schwemmle@hydrology.uni-freiburg.de',

    license='GPLv3',

    classifiers=[
        'Development Status :: 1 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Modelling',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires=">=3.6",
    packages=find_packages(exclude=['docs']),
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    extras_require={
        'test': TEST_REQUIRES + INSTALL_REQUIRES,
    },
)
