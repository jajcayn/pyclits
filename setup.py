"""
created on Sep 22, 2017

@author: Nikola Jajcay, jajcay(at)cs.cas.cz
"""

from setuptools import setup

def readme():
    with open("README.rst") as f:
        return f.read()

setup(name = "pyclits",
    version = "0.1",
    description = "Python Climate Time Series package",
    long_description = readme(),
    classifiers = [
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License"
    ],
    keywords = "time series analysis climate data",
    url = "https://github.com/jajcayn/pyclits",
    author = "Nikola Jajcay",
    author_email = "jajcay@cs.cas.cz",
    license = "MIT",
    packages = ["pyclits"],
    install_requires = [
        "numpy",
        "scipy",
        "scikit-learn",
        "cython",
        "matplotlib",
        "netCDF4"
    ],
    include_package_data = True,
    zip_safe = False)