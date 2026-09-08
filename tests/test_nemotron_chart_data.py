import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('nemotron_data',Path(__file__).parents[1]/'scripts/prepare_nemotron_chart_data.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_data(tmp_path, leak=False):
    source=tmp_path/'source';source.mkdir()
    for split in ('train','validation','test'):
        row={'id': split,'group': 'shared' if leak else split,'split':split,'images':[],
             'messages':[{'role':'user','content':'Token amount?'},{'role':'assistant','content':'123456789012345678'}], 'sources':['fixture']}
        rows=[row]
        if split=='train': rows.append(dict(row,id='visual',group='visual',images=['images/chart.png']))
        (source/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    return source


def test_preserves_splits_amounts_and_visual_rows(tmp_path):
    source=source_data(tmp_path);out=tmp_path/'out';r=module.prepare(source,out)
    assert r['counts']['train']=={'text_ready':1,'visual_pending':1}
    row=json.loads((out/'train.jsonl').read_text())
    assert row['messages'][1]['content']=='123456789012345678'
    assert json.loads((out/'train-visual-pending.jsonl').read_text())['images']==['images/chart.png']
    assert json.loads((out/'train-provenance.jsonl').read_text())['split']=='train'
    assert r['training_completed'] is False
    with pytest.raises(ValueError,match='new output'):module.prepare(source,out)


def test_rejects_cross_split_family_leakage_before_writing(tmp_path):
    source=source_data(tmp_path,leak=True);out=tmp_path/'out'
    with pytest.raises(ValueError,match='Cross-split group'):module.prepare(source,out)
    assert not out.exists()
