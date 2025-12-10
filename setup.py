# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

"""
Core library deps
"""
REQUIRES = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scipy>=1.11.0",
    "scikit-learn>=1.8.0",
    "ipython",
    "patsy",
    "seaborn",
    "plotly",
    "matplotlib",
    "statsmodels",
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
        keywords=["balance", "statistics", "bias"],
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
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
            "Programming Language :: Python :: 3.14",
            "License :: OSI Approved :: MIT License",
        ],
    )


if __name__ == "__main__":
    setup_package()
