# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path


@unittest.skipUnless(
    files("balance").joinpath("run_balance_tests.sh").is_file(),
    "The Bento test runner is Meta-internal and is not shipped in the OSS package",
)
class BentoRunnerScopeTest(unittest.TestCase):
    def _pytest_arguments(self, *arguments: str) -> tuple[list[str], str]:
        with tempfile.TemporaryDirectory() as temporary, as_file(
            files("balance").joinpath("run_balance_tests.sh")
        ) as runner:
            home = Path(temporary)
            log = home / "argv.json"
            kernel = home / "bento_kernel_core_stats_binary-inplace#link-tree"
            kernel.mkdir()
            bootstrap = kernel / "_bootstrap.sh"
            program = (
                "import json,os,sys; "
                "open(os.environ['TEST_ARGV_LOG'],'w').write(json.dumps(sys.argv[1:])); "
                "print('captured pytest arguments')"
            )
            bootstrap.write_text(
                f'#!/bin/bash\n{shlex.quote(sys.executable)} -c {shlex.quote(program)} "$@"\n'
            )
            bootstrap.chmod(0o700)
            subprocess.run(
                ["bash", str(runner), *arguments],
                env={**os.environ, "HOME": str(home), "TEST_ARGV_LOG": str(log)},
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            argv = json.loads(log.read_text())
            return argv[2:], str(runner.parent / "parent_balance" / "tests")

    def test_flag_only_run_includes_package_test_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, as_file(
            files("balance").joinpath("run_balance_tests.sh")
        ) as runner:
            home = Path(temporary)
            log = home / "argv.json"
            kernel = home / "bento_kernel_core_stats_binary-inplace#link-tree"
            kernel.mkdir()
            program = "import json,os,sys; open(os.environ['TEST_ARGV_LOG'],'w').write(json.dumps(sys.argv[1:])); print('captured pytest arguments')"
            bootstrap = kernel / "_bootstrap.sh"
            bootstrap.write_text(
                f'#!/bin/bash\n{shlex.quote(sys.executable)} -c {shlex.quote(program)} "$@"\n'
            )
            bootstrap.chmod(0o700)
            subprocess.run(
                ["bash", str(runner), "-q"],
                env={**os.environ, "HOME": str(home), "TEST_ARGV_LOG": str(log)},
                cwd=temporary,
                check=True,
                capture_output=True,
                text=True,
            )
            argv = json.loads(log.read_text())
            self.assertEqual(argv[0], "-c")
            tests = str(runner.parent / "parent_balance/tests")
            self.assertEqual(argv[2:], [tests, "-q", "-v", "--tb=short"])

    def test_flag_before_file_keeps_only_the_selected_test(self) -> None:
        arguments, tests = self._pytest_arguments("-q", "test_sample.py")

        self.assertEqual(arguments[:2], ["-q", tests + "/test_sample.py"])
        self.assertNotIn(tests, arguments)

    def test_flag_before_python_path_keeps_only_the_selected_path(self) -> None:
        arguments, tests = self._pytest_arguments("-q", "tests/test_foo.py")

        self.assertEqual(arguments[:2], ["-q", "tests/test_foo.py"])
        self.assertNotIn(tests, arguments)

    def test_flag_before_directory_keeps_only_the_selected_directory(self) -> None:
        directory = tempfile.gettempdir()
        arguments, tests = self._pytest_arguments("-q", directory)

        self.assertEqual(arguments[:2], ["-q", directory])
        self.assertNotIn(tests, arguments)

    def test_deselect_value_keeps_package_scope_and_is_not_prefixed(self) -> None:
        arguments, tests = self._pytest_arguments(
            "--deselect", "test_sample.py::SampleTest::test_x"
        )

        self.assertEqual(
            arguments[:3], [tests, "--deselect", "test_sample.py::SampleTest::test_x"]
        )

    def test_flag_values_cannot_replace_package_test_directory(self) -> None:
        arguments, tests = self._pytest_arguments("--basetemp", "/tmp", "-q")
        self.assertEqual(arguments[:3], [tests, "--basetemp", "/tmp"])

    def test_explicit_test_file_stays_selected(self) -> None:
        arguments, _ = self._pytest_arguments("test_sample.py", "-k", "from_frame")
        self.assertEqual(Path(arguments[0]).name, "test_sample.py")
        self.assertEqual(arguments[1:3], ["-k", "from_frame"])

    def test_default_run_is_scoped(self) -> None:
        arguments, tests = self._pytest_arguments()
        self.assertEqual(arguments[0], tests)
