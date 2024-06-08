# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed according to the terms of the
# GNU General Public License version 2.


from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import balance
import balance.testutil


class TestBalanceSetWarnings(balance.testutil.BalanceTestCase):
    def test_balance_set_warnings(self):
        logger = logging.getLogger(__package__)

        balance.set_warnings("WARNING")
        self.assertNotWarns(logger.debug, "test_message")

        balance.set_warnings("DEBUG")
        self.assertWarnsRegexp("test_message", logger.debug, "test_message")
        self.assertWarnsRegexp("test_message", logger.warning, "test_message")

        balance.set_warnings("WARNING")
        self.assertWarnsRegexp("test_message", logger.warning, "test_message")
        self.assertNotWarns(logger.debug, "test_message")
