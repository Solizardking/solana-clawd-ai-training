"""Copy the official static Space at a pinned Hub revision for deployment."""
import json
from pathlib import Path
from urllib.request import urlopen

SPACE = "solanaclawd/clawd-ai"
ROOT = Path(__file__).resolve().parent

def main():
    with urlopen(f"https://huggingface.co/api/spaces/{SPACE}", timeout=30) as response:
        metadata = json.load(response)
    revision = metadata["sha"]
    output = ROOT / "public"
    output.mkdir(exist_ok=True)
    for name in ("index.html", "style.css"):
        with urlopen(f"https://huggingface.co/spaces/{SPACE}/resolve/{revision}/{name}", timeout=30) as response:
            content = response.read()
        if name.endswith(".html") and b"<!doctype html>" not in content.lower():
            raise ValueError("Expected the Space HTML")
        (output / name).write_bytes(content)
    (ROOT / "source.json").write_text(json.dumps({"space": SPACE, "revision": revision}, indent=2) + "\n")
    print(f"Synced {SPACE} at {revision}")

if __name__ == "__main__":
    main()
