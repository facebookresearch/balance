# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

"""
Core library deps

Version requirements
* numpy: Using latest available numpy version to avoid binary compatibility issues
* pandas<=2.0.3: Newer versions lead to "AttributeError: module 'pandas.core.arrays.numpy_' has no attribute 'PandasArray'"
  and numpy binary incompatibility issues. Using 2.0.3 for all Python versions ensures consistent behavior.
* scipy<=1.10.1 and scikit-learn<=1.2.2: Necessary for numerical tests to pass for Python < 3.12. May be possible to relax these without major issues.
* scipy>=1.11.0 and scikit-learn>=1.3.0: Required for Python 3.12 compatibility.
"""
REQUIRES = [
    "numpy",
    "pandas<=2.0.3",
    "ipython",
    "scipy<=1.10.1; python_version<'3.12'",
    "scipy>=1.11.0; python_version>='3.12'",
    "patsy",
    "seaborn",
    "plotly",
    "matplotlib",
    "statsmodels",
    "scikit-learn<=1.2.2; python_version<'3.12'",
    "scikit-learn>=1.3.0; python_version>='3.12'",
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
        license="MIT",
        url="https://github.com/facebookresearch/balance",
        keywords=[""],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">=3.9",
        install_requires=REQUIRES,
        packages=find_packages(include=["balance*"]),
        # Include all csv files
        package_data={"": ["*.csv"]},
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
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: MIT License",
        ],
    )


if __name__ == "__main__":
    setup_package()
