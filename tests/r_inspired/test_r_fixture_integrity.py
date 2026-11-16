# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path


class RFixtureIntegrityTest(unittest.TestCase):
    def test_present_fixture_banks_and_notices_match_their_manifests(self) -> None:
        root = Path(__file__).parent
        self.assertTrue((root / "README.md").is_file())
        for path in sorted((root / "fixtures").glob("*/manifest.json")):
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertTrue(manifest["fixture_sha256"])
            self.assertTrue(manifest["license_sha256"])
            for category in ("fixture_sha256", "license_sha256"):
                for name, expected in manifest[category].items():
                    with self.subTest(
                        bank=path.parent.name, category=category, resource=name
                    ):
                        self.assertEqual(
                            hashlib.sha256(
                                (path.parent / name).read_bytes()
                            ).hexdigest(),
                            expected,
                        )
