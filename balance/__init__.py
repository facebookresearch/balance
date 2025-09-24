# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe


import logging
from typing import Optional

from balance.balancedf_class import (  # noqa
    BalanceCovarsDF,  # noqa
    BalanceOutcomesDF,  # noqa
    BalanceWeightsDF,  # noqa
)
from balance.datasets import load_data  # noqa
from balance.sample_class import Sample  # noqa
from balance.util import TruncationFormatter  # noqa

# TODO: which objects do we want to explicitly externalize?
# TODO: verify this works.

global __version__
__version__ = "0.11.0"


def setup_logging(
    logger_name: Optional[str] = __package__,
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


# TODO: add example in the notebooks for using this function.
def set_warnings(level: str = "WARNING") -> None:
    logger.setLevel(getattr(logging, level))
