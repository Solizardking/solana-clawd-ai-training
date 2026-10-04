#!/usr/bin/env python3
"""Build a history-free, fail-closed public snapshot. Never prints secret values."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/public_release_policy.json"
SENSITIVE_NAMES = re.compile(
    r"(?i)(?:^|/)(?:wallet|keypair|credentials|service-account[^/]*|"
    r"firebase-adminsdk[^/]*|phantom-export|id|auth-token[^/]*|"
    r"access-token[^/]*|refresh-token[^/]*)\.json$"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")
# Standard container/service accounts are not personal workstation identities.
HOME_PATH = re.compile(r"/(?:Users|home)/(?!(?:user|ubuntu|runner|app|nvidia|jovyan)/)[A-Za-z0-9_.-]+/")
WALLET_ARRAY = re.compile(r"\[\s*(?:\d{1,3}\s*,\s*){63}\d{1,3}\s*\]")
TOKEN_QUERY = re.compile(
    r"(?i)https?://[^\s\"'<>]+[?&](?:api[_-]?key|token|secret|access_token)="
    r"([A-Za-z0-9_.-]{12,})"
)
PLACEHOLDERS = {"your_api_key", "your-api-key", "your_token_here", "replace_me", "placeholder"}


def excluded(path: str, policy: dict) -> bool:
    if Path(path).name in policy["allowed_environment_examples"]:
        return False
    if SENSITIVE_NAMES.search(path):
        return True
    if Path(path).name.startswith("config.local.") or path.endswith(".local.js"):
        return True
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in policy["excluded_patterns"])


def inspect_file(path: Path, relative: str) -> list[dict]:
    """Supplement provider token scanning with private state and execution checks."""
    findings = []
    if path.is_symlink():
        return [{"file": relative, "rule": "symlink-not-permitted"}]
    raw = path.read_bytes()
    if relative.endswith(".ipynb"):
        try:
            notebook = json.loads(raw)
            if any(c.get("outputs") or c.get("execution_count") is not None
                   for c in notebook.get("cells", []) if c.get("cell_type") == "code"):
                findings.append({"file": relative, "rule": "notebook-execution-state"})
            if notebook.get("metadata", {}).get("widgets"):
                findings.append({"file": relative, "rule": "notebook-widget-state"})
        except (ValueError, AttributeError):
            findings.append({"file": relative, "rule": "invalid-notebook"})
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        # Non-text files are still scanned by gitleaks. No claim of PDF/image OCR.
        return findings
    for rule, pattern in [("private-key", PRIVATE_KEY), ("personal-home-path", HOME_PATH)]:
        for match in pattern.finditer(content):
            findings.append({"file": relative, "rule": rule,
                             "line": content.count("\n", 0, match.start()) + 1})
    for match in WALLET_ARRAY.finditer(content):
        values = json.loads(match.group())
        if all(0 <= value <= 255 for value in values):
            findings.append({"file": relative, "rule": "possible-solana-secret-key-array",
                             "line": content.count("\n", 0, match.start()) + 1})
    for match in TOKEN_QUERY.finditer(content):
        if match.group(1).lower() not in PLACEHOLDERS:
            findings.append({"file": relative, "rule": "credential-in-url",
                             "line": content.count("\n", 0, match.start()) + 1})
    return findings


def inspect_tree(root: Path, policy: dict) -> list[dict]:
    findings = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append({"file": relative, "rule": "symlink-not-permitted"})
        elif path.is_file():
            if excluded(relative, policy):
                findings.append({"file": relative, "rule": "excluded-path"})
            findings.extend(inspect_file(path, relative))
    for required in policy["required_license_files"]:
        if not (root / required).is_file():
            findings.append({"file": required, "rule": "missing-license-or-notice"})
    return findings


def gitleaks(root: Path, report: Path, history: bool = False) -> list[dict]:
    executable = shutil.which("gitleaks")
    if executable is None:
        raise RuntimeError("gitleaks is required; install it before building a public release")
    command = [executable, "git" if history else "dir", str(root), "--redact=100",
               "--ignore-gitleaks-allow", "--report-format=json", f"--report-path={report}",
               "--no-banner", "--max-decode-depth=5"]
    if history:
        command.append("--log-opts=--all")
    # Never let repository configuration suppress provider rules or findings.
    # An explicit empty configuration extends the scanner's default rules.
    configuration = report.with_suffix(".toml")
    configuration.write_text(
        "[extend]\nuseDefault = true\n\n"
        "[[allowlists]]\ndescription = 'Official public CLAWD token mint'\n"
        "regexTarget = 'secret'\n"
        "regexes = ['^8cHzQHUS2s2h8TzCmfqPKYiM4dSt4roa3n7MyRLApump$']\n"
    )
    ignore = report.with_suffix(".ignore")
    ignore.write_text("")
    command.extend([f"--config={configuration}", f"--gitleaks-ignore-path={ignore}"])
    result = subprocess.run(command, capture_output=True, env={
        k: v for k, v in os.environ.items()
        if k not in {"GITLEAKS_CONFIG", "GITLEAKS_CONFIG_TOML"}
    })
    if result.returncode not in (0, 1) or not report.exists():
        raise RuntimeError("gitleaks did not complete; release stopped (scanner output withheld)")
    detected = json.loads(report.read_text())
    return [{"file": f["File"], "rule": f["RuleID"], "line": f["StartLine"],
             **({"commit": f["Commit"]} if history else {})} for f in detected]


def candidate_paths(root: Path) -> list[str]:
    result = subprocess.run(["git", "ls-files", "--cached", "-z"], cwd=root,
                            capture_output=True, check=True)
    return sorted(set(p for p in result.stdout.decode().split("\0") if p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail on excluded files still in Git index")
    parser.add_argument("--history", action="store_true", help="Also require clean reachable Git history")
    parser.add_argument("--output", type=Path, help="Create a new checked directory without .git")
    parser.add_argument("--report", type=Path, default=ROOT / "local/open-source-audit/release-check.json")
    args = parser.parse_args()
    if not args.check and not args.output:
        parser.error("choose --check and/or --output")
    policy = json.loads(POLICY.read_text())
    args.report = args.report.resolve()
    # Reports contain local paths and belong outside the public candidate.
    try:
        args.report.relative_to(ROOT)
    except ValueError:
        pass
    else:
        if not args.report.is_relative_to(ROOT / "local"):
            parser.error("within this checkout, audit reports must be under local/")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(args.report.parent, 0o700)
    if args.output:
        args.output = args.output.resolve()
        if args.output.exists():
            parser.error("output already exists; choose a new directory")
        if args.output.is_relative_to(ROOT) and not args.output.is_relative_to(ROOT / "local"):
            parser.error("release output must be outside the checkout or under ignored local/")
    paths = candidate_paths(ROOT)
    omitted = [p for p in paths if excluded(p, policy)]
    findings = [{"file": p, "rule": "excluded-path-in-index"} for p in omitted] if args.check else []
    try:
        with tempfile.TemporaryDirectory(prefix="clawd-public-review-") as temporary:
            stage = Path(temporary) / "source"
            stage.mkdir()
            for relative in paths:
                if relative in omitted:
                    continue
                source = ROOT / relative
                if source.is_symlink():
                    findings.append({"file": relative, "rule": "symlink-not-permitted"})
                elif not source.is_file():
                    findings.append({"file": relative, "rule": "missing-indexed-file"})
                else:
                    target = stage / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target)
            findings.extend(inspect_tree(stage, policy))
            findings.extend(gitleaks(stage, args.report.with_name("candidate-gitleaks.json")))
            if args.history:
                findings.extend(gitleaks(ROOT, args.report.with_name("history-gitleaks.json"), True))
            report = {"status": "blocked" if findings else "passed", "files": len(paths) - len(omitted),
                      "excluded": omitted, "history_checked": args.history, "findings": findings,
                      "limitations": ["Pattern scanning is not proof of ownership or absence of every secret.",
                                       "Binary images/PDFs are not OCR audited.",
                                       "A clean export does not sanitize existing Git or remote history."]}
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            os.chmod(args.report, 0o600)
            if findings:
                print(f"BLOCKED: {len(findings)} findings; private report: {args.report}")
                return 1
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(stage, args.output)
            print(f"PASS: {report['files']} files; history checked: {args.history}")
            if args.output:
                print(f"History-free public candidate: {args.output}")
            return 0
    except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"BLOCKED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
