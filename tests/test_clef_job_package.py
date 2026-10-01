"""The submitted package must use expanded data and the same audited code."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from clef_research_data import training_source_hash
from launch_clef_research_job import build_bundle, published_revision
from train_clef_research import select_cohort


def test_remote_bundle_has_identical_code_provenance_and_imports(tmp_path):
    script = tmp_path / "job.py"
    build_bundle(script)
    parsed = ast.parse(script.read_text())
    encoded = next(node.args[0].value for node in ast.walk(parsed)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "b64decode")
    remote = tmp_path / "remote"
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded))) as package:
        package.extractall(remote)
    result = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); from clef_research_data import training_source_hash; import train_clef_research; import clef_live_tape; import export_clef_release; print(training_source_hash())",
                             str(remote / "scripts")], check=True, text=True, capture_output=True)
    assert result.stdout.strip() == training_source_hash()


def test_pilot_contains_live_observations_both_papers_and_samples_expanded_tail():
    rows = [{"id": str(index), "provenance": {"source_type": "parquet", "source": f"source-{index}"}} for index in range(200)]
    rows += [{"id": "red", "provenance": {"source_type": "pdf", "source": "2605.12151v2.pdf"}},
             {"id": "hour", "provenance": {"source_type": "pdf", "source": "2606.08232v1.pdf"}},
             {"id": "live", "provenance": {"source_type": "live_tape_observation", "source": "wss://clawd-ws.fly.dev/ws"}}]
    selected = select_cohort(rows, 16, 42, include_live=True)
    assert len(selected) == 16
    assert {"red", "hour", "live"} <= {row["id"] for row in selected}
    assert any(int(row["id"]) > 100 for row in selected if row["id"].isdigit())
    assert selected == select_cohort(rows, 16, 42, include_live=True)
    assert select_cohort(rows, 0, 42) is rows
    with pytest.raises(ValueError, match="required live/paper"):
        select_cohort(rows, 2, 42, include_live=True)


def test_publication_must_verify_entire_dataset_package(tmp_path):
    from expand_realtime_research import REPO, refresh_package
    (tmp_path / "metadata").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "README.md").write_text("https://arxiv.org/abs/2605.12151 https://arxiv.org/abs/2606.08232")
    (tmp_path / "metadata/expansion_lineage.jsonl").write_text("")
    outputs = {}
    import pyarrow as arrow
    import pyarrow.parquet as parquet
    for split in ("train", "eval", "test"):
        filename = f"data/{split}-00000-of-00001.parquet"
        parquet.write_table(arrow.table({"messages": [[{"role": "user", "content": split}]]}), tmp_path / filename)
        outputs[split] = {"rows": 1, "file": filename}
    manifest = {"repo_id": REPO, "base_revision": "1" * 40, "outputs": outputs,
                "original_rows_preserved_in_original_splits": True}
    (tmp_path / "metadata/realtime_research_expansion_manifest.json").write_text(json.dumps(manifest))
    refresh_package(tmp_path)
    package = json.loads((tmp_path / "package.json").read_text())
    publication = tmp_path / "published.json"
    result = {"commit": "2" * 40, "verified_files": [entry["path"] for entry in package["files"]],
              "split_rows": {split: 1 for split in outputs}}
    publication.write_text(json.dumps(result))
    assert published_revision(publication) == "2" * 40
    result["verified_files"].pop()
    publication.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="every staged file"):
        published_revision(publication)
