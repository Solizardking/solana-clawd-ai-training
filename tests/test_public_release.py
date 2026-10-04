"""Release tests protect boundaries rather than assert implementation details."""
import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_release", ROOT / "scripts/prepare_public_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
POLICY = json.loads((ROOT / "configs/public_release_policy.json").read_text())


class PublicReleaseTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gitleaks"), "requires the release secret scanner")
    def test_real_scanner_blocks_synthetic_credential_and_redacts_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            token = "hf_" + hashlib.sha256(b"synthetic-public-release-test").hexdigest()[:34]
            (source / "config.txt").write_text("HF_TOKEN=" + token + "\n")
            report = root / "findings.json"
            self.assertTrue(release.gitleaks(source, report))
            self.assertNotIn(token, report.read_text())

    def test_private_material_and_fonts_are_excluded(self):
        paths = [".work/gateway.py", "artifacts/paper.pdf", "data/nested/train.jsonl",
                 ".venv-connect/lib/file.py", "nested/node_modules/lib.js",
                 "wallet.json", "nested/keypair.json", "nested/.env.production",
                 "nested/brand/font.otf", "memory/run.json", "state.ses",
                 "nested/config.local.js", "outputs/checkpoint/model.safetensors"]
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(release.excluded(path, POLICY))
        self.assertFalse(release.excluded("chart_agent/credentials.py", POLICY))
        self.assertFalse(release.excluded("memory/honcho.py", POLICY))
        self.assertFalse(release.excluded("tests/fixtures/clawd_ws/token-launch.json", POLICY))
        self.assertFalse(release.excluded("nested/.env.example", POLICY))

    def test_wallet_array_and_private_key_fail_without_printing_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "innocent.txt"
            path.write_text(json.dumps(list(range(64))) + "\n" +
                            "-----BEGIN " + "PRIVATE KEY-----\n")
            findings = release.inspect_file(path, "innocent.txt")
            rules = {f["rule"] for f in findings}
            self.assertIn("private-key", rules)
            self.assertIn("possible-solana-secret-key-array", rules)
            self.assertTrue(all(set(f) <= {"file", "rule", "line"} for f in findings))

    def test_symlink_does_not_copy_private_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "public"
            root.mkdir()
            private = Path(tmp) / "private"
            private.write_text("must stay outside export")
            link = root / "data.txt"
            link.symlink_to(private)
            findings = release.inspect_tree(root, {**POLICY, "required_license_files": []})
            self.assertEqual(findings, [{"file": "data.txt", "rule": "symlink-not-permitted"}])

    def test_notebook_outputs_and_missing_notices_block_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            notebook = root / "example.ipynb"
            notebook.write_text(json.dumps({"cells": [{"cell_type": "code", "outputs":
                [{"text": "private runtime output"}], "execution_count": 1}], "metadata": {}}))
            findings = release.inspect_tree(root, POLICY)
            self.assertIn("notebook-execution-state", {f["rule"] for f in findings})
            self.assertEqual(sum(f["rule"] == "missing-license-or-notice" for f in findings),
                             len(POLICY["required_license_files"]))

    def test_unknown_url_credentials_and_personal_paths_are_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "example.txt"
            path.write_text("https://example.invalid/?api_key=" + "x" * 20 +
                            "\n" + "/Users/" + "private-person" + "/work")
            self.assertEqual({f["rule"] for f in release.inspect_file(path, "example.txt")},
                             {"credential-in-url", "personal-home-path"})


if __name__ == "__main__":
    unittest.main()
