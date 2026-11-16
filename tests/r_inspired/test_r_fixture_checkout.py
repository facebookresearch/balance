# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


def fixture_digests(fixtures: Path) -> dict[Path, str]:
    resources: dict[Path, str] = {}
    for bank in sorted(fixtures.iterdir()):
        manifest = bank / "manifest.json"
        if not bank.is_dir() or not manifest.is_file():
            continue
        data = json.loads(manifest.read_text())
        for category in ("fixture_sha256", "license_sha256"):
            for name, digest in data[category].items():
                resources[Path(bank.name) / name] = digest
    return resources


def checkout_index(root: Path, autocrlf: str) -> None:
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    for args in (
        ("init", "--quiet"),
        ("add", "fixtures"),
        ("checkout-index", "--all", "--prefix=checkout/"),
    ):
        subprocess.run(
            [
                "git",
                "-c",
                f"core.autocrlf={autocrlf}",
                "-c",
                "core.safecrlf=false",
                "-c",
                f"core.attributesFile={os.devnull}",
                *args,
            ],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            timeout=30,
        )


class FixtureCheckoutTest(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("git"), "requires Git to exercise checkout conversion"
    )
    def test_git_checkout_preserves_exact_fixture_bytes(self) -> None:
        fixtures = Path(__file__).resolve().parent / "fixtures"
        resources = fixture_digests(fixtures)
        self.assertTrue(resources)
        for autocrlf in ("true", "false"):
            with self.subTest(autocrlf=autocrlf), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                destination = root / "fixtures"
                destination.mkdir()
                shutil.copyfile(
                    fixtures / ".gitattributes", destination / ".gitattributes"
                )
                for relative in resources:
                    file = destination / relative
                    file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(fixtures / relative, file)
                crlf = destination / "original-crlf.txt"
                crlf.write_bytes(b"original\r\nbytes\r\n")
                checkout_index(root, autocrlf)
                checked_out = root / "checkout/fixtures"
                for relative, expected in resources.items():
                    with self.subTest(resource=str(relative)):
                        self.assertEqual(
                            hashlib.sha256(
                                (checked_out / relative).read_bytes()
                            ).hexdigest(),
                            expected,
                        )
                self.assertEqual(
                    (checked_out / crlf.name).read_bytes(), b"original\r\nbytes\r\n"
                )
