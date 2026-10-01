#!/usr/bin/env python3
"""Stage public-safe source documentation without treating it as new training rows.

The inventories in ``data`` describe several historical datasets. Their counts
are documentation; only an ingestion step that reads an actual payload can
claim new examples. This module deliberately stages Markdown and metadata JSON,
not the raw corpora, generated caches, credentials, or binary indexes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import unquote


REDACTED_SECRET = "[REDACTED_SECRET]"
LOCAL_PATH = "[LOCAL_PATH]"

# Match actual credential payloads, not documentation such as hf_... or $HF_TOKEN.
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP |DSA )?PRIVATE KEY-----"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bwandb(?:_v1)?_[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
)

_KEY_NAMES = (
    r"api[_-]?key|private[_-]?key|signing[_-]?key|client[_-]?secret|secret|"
    r"(?:secret[_-]?)?access[_-]?token|refresh[_-]?token|id[_-]?token|"
    r"(?:aws[_-]?)?secret[_-]?access[_-]?key|password|passwd|"
    r"(?:hf|hf_hub|github|openai|anthropic|nvidia|wandb)[_-]?(?:token|key|api[_-]?key)|"
    r"(?:[A-Za-z][A-Za-z0-9]*[_-])+(?:api[_-]?key|access[_-]?token|private[_-]?key|client[_-]?secret|password)|"
    r"_authToken|authorization"
)
_SENSITIVE_KEY = re.compile(rf"^(?:{_KEY_NAMES})$", re.IGNORECASE)
_QUOTED_ASSIGNMENT = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9_])(?:{_KEY_NAMES})\b(?:\\*[\"'])?\s*[:=]\s*)"
    r"(?P<quote>\\*[\"'])(?P<value>[^\r\n]*?)(?P=quote)",
    re.IGNORECASE,
)
_ASSIGNMENT = re.compile(
    rf"(?P<prefix>(?<![A-Za-z0-9_])(?:{_KEY_NAMES})\b(?:\\*[\"'])?\s*[:=]\s*)"
    r"(?P<quote>[\"']?)(?P<value>[^\\\s\"'`,;&?#<>\[\]{}]+)(?P=quote)",
    re.IGNORECASE,
)
_BEARER = re.compile(r"(?P<prefix>\b(?:Bearer|Basic)\s+)(?P<value>[^\s\"'`,;<>\[\]{}]+)", re.IGNORECASE)
_URL_AUTH_QUERY = re.compile(
    r"(?P<prefix>(?:[?&]|&amp;)(?:api[_-]?key|key|token|access[_-]?token|refresh[_-]?token|"
    r"auth|authorization|secret|password|client[_-]?secret|"
    r"x-amz-(?:credential|signature|security-token)|x-goog-(?:credential|signature))=)"
    r"(?P<value>[^&#\s\"'`<>\[\]{}]+)",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(?P<prefix>https?://)(?P<value>[^/\s\"'`<>]+:[^/\s\"'`<>]+)@", re.IGNORECASE)
_KEY_ARRAY = re.compile(
    r"(?P<prefix>\b(?:private[_-]?key|signing[_-]?key|keypair)[\"']?\s*[:=]\s*)"
    r"(?P<value>\[(?:\s*\d{1,3}\s*,){31,}\s*\d{1,3}\s*\])",
    re.IGNORECASE,
)

# Replace local roots, retaining the useful source suffix. Match encoded paths
# too; public URL paths such as arxiv.org/abs/... are unaffected.
_LOCAL_PATH_PATTERNS = (
    re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^/\s\"'`<>\[\]{}(),]+", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])/(?:root|workspace|workspaces|tmp|private|var|mnt|Volumes|opt|etc|srv)(?=/|\s|[\"'`<>\[\]{}(),]|$)", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:%2f)(?:Users|home)(?:%2f)[^%\s\"'`<>\[\]{}(),]+", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/])(?:Users[\\/][^\\/\s\"'`<>\[\]{}(),]+)?", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9])~(?=/|\\|[A-Za-z0-9_-]+/)", re.IGNORECASE),
)


def _placeholder(value: Any) -> bool:
    """Keep environment references, markers, and deliberately illustrative keys."""
    if value is None or isinstance(value, (bool, int, float)):
        return True
    if not isinstance(value, str):
        return False
    decoded = unquote(value).strip().strip("\"'")
    if decoded.lower().startswith(("bearer ", "basic ")):
        return _placeholder(decoded.split(None, 1)[1])
    if not decoded:
        return True
    if decoded.startswith(("$", "[REDACTED", "[LOCAL_PATH]")):
        return True
    if decoded.startswith("<") and decoded.endswith(">"):
        return True
    upper = decoded.upper()
    if upper in {"NONE", "NULL", "TRUE", "FALSE", "TOKEN", "KEY", "SECRET", "PASSWORD", "API_KEY", "HF_TOKEN", "BEARER", "BASIC", "YOUR_TOKEN", "YOUR_API_KEY", "YOUR_KEY", "YOUR_SECRET", "REPLACE_ME", "REPLACE-ME", "CHANGEME", "PLACEHOLDER", "EXAMPLE", "REDACTED", "...", "…"}:
        return True
    if re.fullmatch(r"(?:YOUR|EXAMPLE|DUMMY|PLACEHOLDER)[_-][A-Z0-9_-]+", upper):
        return True
    if decoded.endswith(("...", "…")):
        return True
    return False


def _actual_secret(value: Any) -> bool:
    if _placeholder(value):
        return False
    if isinstance(value, (dict, list, tuple)):
        return bool(value)
    # Sensitive assignments name the purpose, so short real passwords count.
    return isinstance(value, str) and bool(value.strip())


def secret_like(text: str) -> bool:
    """Identify credential-bearing text for rejecting candidate chat examples."""
    if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
        return True
    for pattern in (_QUOTED_ASSIGNMENT, _ASSIGNMENT, _BEARER, _URL_AUTH_QUERY, _URL_USERINFO, _KEY_ARRAY):
        if any(_actual_secret(match.group("value")) for match in pattern.finditer(text)):
            return True
    return False


def local_path_like(text: str) -> bool:
    return any(pattern.search(text) for pattern in _LOCAL_PATH_PATTERNS)


def _sanitize_string(text: str) -> str:
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED_SECRET, text)

    def redact_value(match: re.Match[str]) -> str:
        if not _actual_secret(match.group("value")):
            return match.group(0)
        quote = match.groupdict().get("quote") or ""
        return match.group("prefix") + quote + REDACTED_SECRET + quote

    for pattern in (_QUOTED_ASSIGNMENT, _ASSIGNMENT, _BEARER, _URL_AUTH_QUERY, _KEY_ARRAY):
        text = pattern.sub(redact_value, text)

    def redact_userinfo(match: re.Match[str]) -> str:
        return match.group("prefix") + REDACTED_SECRET + "@"

    text = _URL_USERINFO.sub(redact_userinfo, text)
    for pattern in _LOCAL_PATH_PATTERNS:
        text = pattern.sub(LOCAL_PATH, text)
    return text


def sanitize_public(value: Any) -> Any:
    """Recursively remove secrets and workstation roots from public artifacts.

    Public keys/mints/signatures are preserved, including strings stored in
    ``public_key``, ``mint``, and ``token`` fields. Sensitive credential field
    names provide the context necessary to scrub unprefixed opaque payloads.
    """
    if isinstance(value, str):
        return _sanitize_string(value)
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            public_key = _sanitize_string(key) if isinstance(key, str) else key
            if isinstance(key, str) and _SENSITIVE_KEY.fullmatch(key) and _actual_secret(item):
                clean[public_key] = REDACTED_SECRET
            else:
                clean[public_key] = sanitize_public(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize_public(item) for item in value]
    return value


_SKIP_DIRS = {".git", ".cache", "cache", "caches", "__pycache__", "node_modules", ".venv", "credentials", "secrets"}
_CREDENTIAL_NAME = re.compile(r"(?:^\.env(?:\.|$)|credentials?|client[_-]?secret|service[_-]?account|keypair|(?:^|[_-])(?:private[_-]?key|access[_-]?token|hf[_-]?token)(?:[_\-.]|$))", re.IGNORECASE)


def _artifact_role(relative: Path) -> str | None:
    name = relative.name.lower()
    parts = {part.lower() for part in relative.parts[:-1]}
    if relative.suffix.lower() == ".md":
        if "citation" in name:
            return "research_citations"
        if "source_notes" in parts:
            return "source_documentation"
        if "eval" in name or "report" in name:
            return "evaluation_documentation"
        return "dataset_documentation"
    if relative.suffix.lower() != ".json":
        return None
    if "manifest" in name or "manifests" in parts:
        return "source_manifest"
    if "report" in name or "eval" in name or "reports" in parts:
        return "evaluation_report"
    if name in {"dataset_info.json", "dataset_dict.json", "state.json"} or parts & {"metadata", "meta"}:
        return "dataset_metadata"
    return None


def _skip_path(relative: Path) -> bool:
    return any(part.lower() in _SKIP_DIRS or part.lower().endswith("_cache") for part in relative.parts[:-1]) or bool(_CREDENTIAL_NAME.search(relative.name))


def stage_supporting_artifacts(data_root: Path, output_root: Path) -> list[dict[str, Any]]:
    """Copy sanitized source docs under ``metadata/local_sources`` and inventory.

    Inventory hashes cover bytes actually read/written, not historic counts.
    Output inside ``data_root`` is excluded to prevent recursive re-ingestion.
    Symlinks are excluded so discovery cannot escape the supplied source root.
    Invalid JSON or residual unsafe text fails staging instead of publishing it.
    """
    data_root = Path(data_root).resolve()
    output_root = Path(output_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Source data directory does not exist: {data_root.name}")
    if data_root == output_root:
        raise ValueError("Output directory must differ from the source data directory")
    inventory: list[dict[str, Any]] = []
    for source in sorted(data_root.rglob("*")):
        if source.is_symlink() or not source.is_file() or source.resolve().is_relative_to(output_root):
            continue
        relative = source.relative_to(data_root)
        role = _artifact_role(relative)
        if role is None or _skip_path(relative):
            continue
        source_bytes = source.read_bytes()
        if b"\x00" in source_bytes:
            continue
        try:
            source_text = source_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if source.suffix.lower() == ".json":
            clean = sanitize_public(json.loads(source_text))
            staged_text = json.dumps(clean, ensure_ascii=False, indent=2) + "\n"
        else:
            staged_text = sanitize_public(source_text)
        if secret_like(staged_text) or local_path_like(staged_text):
            raise ValueError(f"Unsafe content remains in sanitized artifact: {relative.as_posix()}")
        staged_bytes = staged_text.encode("utf-8")
        repo_path = Path("metadata/local_sources") / relative
        destination = output_root / repo_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(staged_bytes)
        inventory.append({
            "source": relative.as_posix(),
            "source_exists": True,
            "source_bytes": len(source_bytes),
            "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "path_in_repo": repo_path.as_posix(),
            "bytes": len(staged_bytes),
            "sha256": hashlib.sha256(staged_bytes).hexdigest(),
            "role": role,
            "new_examples": 0,
            "historical_counts_are_documentation": True,
        })
    return inventory
