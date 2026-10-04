"""Package tracked Model Kit source without credentials, caches or runtimes."""
import re
import subprocess
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[2]
destination = root / "onchainai/downloads/model-kit-source.zip"
destination.parent.mkdir(parents=True, exist_ok=True)
paths = subprocess.check_output(["git", "ls-files", "model-kit"], cwd=root, text=True).splitlines()
secret = re.compile(rb"hf_[A-Za-z0-9]{24,}|(?:sk-|nvapi-)[A-Za-z0-9_-]{20,}|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY")
with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
    for name in paths:
        path = root / name
        if not path.is_file() or any(part.startswith(".env") or part in {".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        contents = path.read_bytes()
        if secret.search(contents):
            raise SystemExit(f"Credential pattern found in {name}; refusing to package")
        archive.writestr(name, contents)
    for name in ["LICENSE", "LICENSING.md", "NOTICE", "THIRD_PARTY_NOTICES.md"]:
        archive.write(root / name, name)
print(f"Packaged {len(paths)} Model Kit source files")
