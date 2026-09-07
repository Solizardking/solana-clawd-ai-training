import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq


def test_preserves_holdouts_and_image_families(tmp_path):
    spec = importlib.util.spec_from_file_location("build_chart", Path(__file__).parents[1] / "scripts/build_chart_foundation_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    data, bucket, papers, output = [tmp_path / name for name in ("data", "local", "papers", "result")]
    for path in (data, bucket / "data", bucket / "images", papers, tmp_path / "outputs/chart-research"):
        path.mkdir(parents=True)
    messages = [{"role": "user", "content": "held out question"}, {"role": "assistant", "content": "held out answer"}]
    (data / "raw.jsonl").write_text(json.dumps({"messages": messages}) + "\n")
    pq.write_table(pa.Table.from_pylist([{"messages": messages}]), data / "eval.parquet")
    (bucket / "images/test.png").write_bytes(b"fixture")
    (bucket / "metadata.csv").write_text("images/test.png,id,synthetic,Q1,A1,bar,lookup,easy,true,train,note\nimages/test.png,id2,synthetic,Q2,A2,bar,lookup,easy,true,test,note\n")
    pq.write_table(pa.Table.from_pylist([{"mint": "fixture", "description": "x" * 100}]), tmp_path / "outputs/chart-research/solarchive-tokens-2020-10.parquet")
    with patch("sys.argv", ["build", "--data", str(data), "--bucket", str(bucket), "--papers", str(papers), "--output", str(output)]):
        module.main()
    rows = {s: [json.loads(line) for line in (output / f"{s}.jsonl").read_text().splitlines()] for s in module.SPLITS}
    assert not rows["train"]
    assert len(rows["validation"]) == 1
    assert len(rows["test"]) == 2
    assert all(row["images"] == ["images/test.png"] for row in rows["test"])
    assert (output / "images/test.png").read_bytes() == b"fixture"
