# Contributing to balance
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Code Requirements

### Coding Style
* 4 spaces for indentation rather than tabs
* 80 character line length

### Linting
Run the linter via `flake8` (`pip install flake8`) from the root of the Ax repository. Note that we have a [custom flake8 configuration](https://github.com/facebookresearch/balance/blob/main/.flake8).

### Static Type Checking
We use [Pyre](https://pyre-check.org/) for static type checking and require code to be fully type annotated.

### Unit testing
We strongly recommend adding unit testing when introducing new code. To run all unit tests, we recommend installing pytest using `pip install pytest` and running `pytest -ra` from the root of the balance repo.

### Documentation
* We require docstrings on all public functions and classes (those not prepended with `_`).
* We use the [Google docstring style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) & use Sphinx to compile API reference documentation.
* Our [website](https://import-balance.org) leverages Docusaurus 2.0 + Sphinx for generating our documentation content.
* To rule out parsing errors, we suggesting [installing sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) and running `make html` from the balance/sphinx folder. Alternatively, you may also try running `./script/make_docs` from the root of the balance repo folder.

## License
By contributing to balance, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
