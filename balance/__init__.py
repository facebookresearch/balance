# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

from balance.balancedf_class import (  # noqa
    BalanceDFCovars,  # noqa
    BalanceDFOutcomes,  # noqa
    BalanceDFWeights,  # noqa
)
from balance.datasets import load_data  # noqa
from balance.sample_class import Sample  # noqa
from balance.util import TruncationFormatter  # noqa

global __version__
__version__ = "0.12.x"

WELCOME_MESSAGE = f"""
Welcome to balance (Version {__version__})!
An open-source Python package for balancing biased data samples.

📖 Documentation: https://import-balance.org/
🛠️ Get Help / Report Issues: https://github.com/facebookresearch/balance/issues/
📄 Citation:
    Sarig, T., Galili, T., & Eilat, R. (2023).
    balance - a Python package for balancing biased data samples.
    https://arxiv.org/abs/2307.06024

Tip: You can access this information at any time with balance.help()
"""


def help() -> None:
    """Display information about documentation, help, and citation."""
    print(WELCOME_MESSAGE)


def setup_logging(
    logger_name: str | None = __package__,
    level: str = "INFO",
    removeHandler: bool = True,
) -> logging.Logger:
    """
    Initiates a nicely formatted logger called "balance", with level "info".
    """
    if removeHandler:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

    logger = logging.getLogger(logger_name)

    logger.setLevel(getattr(logging, level))
    formatter = TruncationFormatter(
        "%(levelname)s (%(asctime)s) [%(module)s/%(funcName)s (line %(lineno)d)]: %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger: logging.Logger = setup_logging()
logger.info(f"Using {__package__} version {__version__}")

# Print the welcome/help/citation message on import
print(WELCOME_MESSAGE)


def set_warnings(level: str = "WARNING") -> None:
    logger.setLevel(getattr(logging, level))
