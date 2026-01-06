# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import logging

import balance.testutil

# TODO: remove the use of balance_util in most cases, and just import the functions to be tested directly
from balance import util as balance_util


class TestUtil(
    balance.testutil.BalanceTestCase,
):
    def test_truncate_text(self) -> None:
        self.assertEqual(
            balance_util._truncate_text("a" * 6, length=5), "a" * 5 + "..."
        )
        self.assertEqual(balance_util._truncate_text("a" * 4, length=5), "a" * 4)
        self.assertEqual(balance_util._truncate_text("a" * 5, length=5), "a" * 5)

    def test__truncate_text(self) -> None:
        """Test _truncate_text with various string lengths."""
        test_cases = [
            # (input_text, length, expected_result, description)
            ("Hello", 10, "Hello", "Short string not truncated"),
            (
                "This is a very long string that needs truncation",
                10,
                "This is a ...",
                "Long string truncated with ellipsis",
            ),
            ("1234567890", 10, "1234567890", "Exact length string not truncated"),
        ]

        for input_text, length, expected_result, description in test_cases:
            with self.subTest(description=description):
                result = balance_util._truncate_text(input_text, length)
                self.assertEqual(result, expected_result)
                if len(input_text) > length:
                    self.assertEqual(len(result), length + 3)  # length + '...'

    def test_TruncationFormatter(self) -> None:
        """Test TruncationFormatter with long and short log messages."""
        formatter = balance_util.TruncationFormatter("%(message)s")
        MAX_MESSAGE_LENGTH = 2000  # TruncationFormatter truncates at 2000 characters
        ELLIPSIS_LENGTH = 3

        test_cases = [
            # (message, should_truncate, description)
            (
                "x" * (MAX_MESSAGE_LENGTH + 1000),
                True,
                "Long message gets truncated",
            ),
            (
                "This is a short message",
                False,
                "Short message remains unchanged",
            ),
        ]

        for message, should_truncate, description in test_cases:
            with self.subTest(description=description):
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=message,
                    args=(),
                    exc_info=None,
                )
                result = formatter.format(record)

                if should_truncate:
                    self.assertEqual(len(result), MAX_MESSAGE_LENGTH + ELLIPSIS_LENGTH)
                    self.assertTrue(result.endswith("..."))
                else:
                    self.assertEqual(result, message)
