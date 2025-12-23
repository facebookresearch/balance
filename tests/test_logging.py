# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals,
)

import logging

import balance
import balance.testutil


class TestBalanceHelp(balance.testutil.BalanceTestCase):
    def test_help_produces_output(self) -> None:
        """Test that help() produces output.

        Validates that the help function outputs a non-empty message.
        This ensures users can access help information.
        """
        # Simply verify that help() produces some output
        self.assertPrints(balance.help)


class TestBalanceSetupLogging(balance.testutil.BalanceTestCase):
    def test_setup_logging_with_defaults(self) -> None:
        """Test setup_logging with default parameters.

        Validates that setup_logging returns a properly configured logger
        with default settings (INFO level, balance package name).
        This is needed to verify default initialization behavior.
        """
        logger = balance.setup_logging()

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.level, logging.INFO)
        self.assertEqual(logger.name, "balance")

    def test_setup_logging_with_custom_logger_name(self) -> None:
        """Test setup_logging with a custom logger name.

        Validates that setup_logging creates a logger with the specified name.
        This is needed to test flexibility in logger configuration.
        """
        custom_name = "test_custom_logger"
        logger = balance.setup_logging(logger_name=custom_name)

        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, custom_name)

    def test_setup_logging_levels(self) -> None:
        """Test setup_logging with various logging levels.

        Validates that the logger can be configured with different levels.
        Uses subTest to test multiple levels in a single test method.
        """
        test_levels = [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ]

        for level_str, expected_level in test_levels:
            with self.subTest(level=level_str):
                logger = balance.setup_logging(level=level_str)
                self.assertEqual(logger.level, expected_level)

    def test_setup_logging_remove_handler_true(self) -> None:
        """Test setup_logging with removeHandler=True.

        Validates that existing root handlers are removed when removeHandler is True.
        This is needed to test clean logger initialization without handler conflicts.
        """
        # Add a handler to root logger
        initial_handler = logging.StreamHandler()
        logging.root.addHandler(initial_handler)
        initial_count = len(logging.root.handlers)

        # Call setup_logging with removeHandler=True (default)
        balance.setup_logging(removeHandler=True)

        # Verify handlers were removed
        self.assertLess(len(logging.root.handlers), initial_count)

    def test_setup_logging_remove_handler_false(self) -> None:
        """Test setup_logging with removeHandler=False.

        Validates that existing root handlers are preserved when removeHandler is False.
        This is needed to test logger initialization without disrupting existing handlers.
        """
        # Add a handler to root logger
        test_handler = logging.StreamHandler()
        logging.root.addHandler(test_handler)
        initial_count = len(logging.root.handlers)

        # Call setup_logging with removeHandler=False
        balance.setup_logging(removeHandler=False)

        # Verify handlers were not removed
        self.assertGreaterEqual(len(logging.root.handlers), initial_count)

        # Cleanup
        logging.root.removeHandler(test_handler)

    def test_setup_logging_formatter_attached(self) -> None:
        """Test that setup_logging attaches a formatter to the handler.

        Validates that the logger's handler has a formatter configured.
        This is needed to ensure log messages are properly formatted.
        """
        logger = balance.setup_logging()

        # Check that logger has handlers
        self.assertGreater(len(logger.handlers), 0)

        # Check that the handler has a formatter
        handler = logger.handlers[0]
        self.assertIsNotNone(handler.formatter)

    def test_setup_logging_returns_correct_logger(self) -> None:
        """Test that setup_logging returns the logger with the correct name.

        Validates that the returned logger matches the requested logger name.
        This is needed to ensure proper logger retrieval and configuration.
        """
        test_name = "test_logger_instance"
        logger = balance.setup_logging(logger_name=test_name)

        # Verify the logger has the correct name
        self.assertEqual(logger.name, test_name)

        # Verify we can retrieve the same logger
        retrieved_logger = logging.getLogger(test_name)
        self.assertEqual(logger, retrieved_logger)

    def test_setup_logging_invalid_level_raises_error(self) -> None:
        """Test that invalid logging levels raise appropriate errors.

        Validates that setup_logging raises AttributeError when given an invalid level.
        This is needed to ensure proper error handling for invalid configuration.
        """
        with self.assertRaises(AttributeError):
            balance.setup_logging(level="INVALID_LEVEL")

    def test_setup_logging_with_none_logger_name(self) -> None:
        """Test setup_logging with None as logger_name.

        Validates that setup_logging handles None logger_name gracefully.
        This is needed to test edge case handling for logger name configuration.
        When None is passed, Python's logging.getLogger(None) returns the root logger.
        """
        logger = balance.setup_logging(logger_name=None)
        self.assertIsInstance(logger, logging.Logger)
        # When logger_name is None, getLogger returns the root logger with empty string name
        self.assertEqual(logger.name, "root")


class TestBalanceSetWarnings(balance.testutil.BalanceTestCase):
    def test_balance_set_warnings(self) -> None:
        logger = logging.getLogger(__package__)

        balance.set_warnings("WARNING")
        self.assertNotWarns(logger.debug, "test_message")

        balance.set_warnings("DEBUG")
        self.assertWarnsRegexp("test_message", logger.debug, "test_message")
        self.assertWarnsRegexp("test_message", logger.warning, "test_message")

        balance.set_warnings("WARNING")
        self.assertWarnsRegexp("test_message", logger.warning, "test_message")
        self.assertNotWarns(logger.debug, "test_message")
