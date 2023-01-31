# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.

from setuptools import find_packages, setup

"""
Core library deps

Version requirements
* scipy<=1.9.2: Required by glmnet-python
* pandas<=1.4.3: 1.5.0 moved UndefinedVariableError into pandas.errors

Other requirements:
* glmnet-python@1.0, from https://github.com/bbalasub1/glmnet_python.git

"""
REQUIRES = [
    "numpy",
    "pandas<=1.4.3",
    "ipython",
    "scipy<=1.9.2",
    "patsy",
    "seaborn<=0.11.1",
    "plotly",
    "matplotlib",
    "statsmodels",
    "scikit-learn",
    "ipfn",
    "session-info",
]

# Development deps (e.g. pytest, builds)
# TODO[scubasteve]: Add dev requirements
DEV_REQUIRES = [
    "setuptools_scm",
    "wheel",
    "pytest",
    "sphinx",
    "notebook",
    "nbconvert",
]

DESCRIPTION = (
    "balance is a Python package offering a simple workflow and methods for "
    "dealing with biased data samples when looking to infer from them to "
    "some target population of interest."
)


def setup_package() -> None:
    """Used for building/installing the balance package."""
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setup(
        name="balance",
        description=DESCRIPTION,
        author="Facebook, Inc.",
        license="GPLv2",
        url="https://github.com/facebookresearch/balance",
        keywords=[""],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">=3.7",
        install_requires=REQUIRES,
        packages=find_packages(include=["balance*"]),
        extras_require={
            "dev": DEV_REQUIRES,
        },
        use_scm_version={
            "write_to": "version.py",
        },
        setup_requires=["setuptools_scm"],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
        ],
    )


if __name__ == "__main__":
    setup_package()
