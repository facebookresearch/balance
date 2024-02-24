# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 2.

import logging
from typing import Optional

from balance.balancedf_class import BalanceCovarsDF, BalanceOutcomesDF, BalanceWeightsDF
from balance.datasets import load_data
from balance.sample_class import Sample
from balance.util import TruncationFormatter

__version__ = "0.9.1"

def setup_logging(
    logger_name: Optional[str] = "balance",
    level: str = "INFO",
    remove_handler: bool = True,
) -> logging.Logger:
    """
    Initiates a nicely formatted logger called "balance", with level "info".
    """
    if remove_handler:
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
logger.info(f"Using balance version {__version__}")

# TODO: Add example in the notebooks for using this function.
def set_warnings(level: str = "WARNING") -> None:
    logger.setLevel(getattr(logging, level))
