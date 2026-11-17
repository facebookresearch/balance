# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
import unittest
from pathlib import Path


class ReleaseWorkflowTest(unittest.TestCase):
    def test_release_jobs_keep_python_39_compatible_runners(self) -> None:
        # GitHub publishes Python 3.9 Linux x64 installers for these Ubuntu versions:
        # https://raw.githubusercontent.com/actions/python-versions/main/versions-manifest.json
        workflow = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "release.yml"
        ).read_text()
        runners = [
            value.strip().strip("\"'")
            for value in re.findall(r"^\s*runs-on:\s*([^#\n]+)", workflow, re.MULTILINE)
        ]
        self.assertTrue(runners)
        unsupported = [
            runner
            for runner in runners
            if runner not in {"ubuntu-22.04", "ubuntu-24.04"}
        ]
        self.assertEqual(unsupported, [])
