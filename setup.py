#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
from skbuild import setup
import setuptools


with open("README.md") as f:
    LONG_DESCRIPTION = f.read()

PACKAGENAME = 'dasilva-invariants'
DESCRIPTION = 'Calculation of Adiabatic Invariants from LFM, T96, and TS05 magnetic field models'
AUTHOR = 'Daniel da Silva'
AUTHOR_EMAIL = 'mail@danieldasilva.org'
LICENSE = 'BSD-3'
URL = 'https://dasilva-invariants.readthedocs.io/en/latest/'
# VERSION should be PEP440 compatible (http://www.python.org/dev/peps/pep-0440)
VERSION = '0.0.1'

setup(
    name=PACKAGENAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    packages=['dasilva_invariants',
              'dasilva_invariants._fortran',
    ],
    long_description=LONG_DESCRIPTION,
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable  ",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    package_data={
        'dasilva_invariants': ['_fortran/*.so'],
    },    
)
