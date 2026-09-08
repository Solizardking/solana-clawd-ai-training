import hashlib
import json
from pathlib import Path
import pytest
from scripts.prepare_reviewed_nemotron_visual import prepare


def fixture(tmp_path):
    source=tmp_path/'source';source.mkdir()
    observation={'size':[100,100],'ocr':{'available':True,'words':[{'text':'$4900'}]},'chart_elements':[]}
    path=source/'observation.json';path.write_text(json.dumps(observation))
    row={'id':'fixture','split':'train','group':'image-one','observations':['observation.json'],
         'original_messages':[{'role':'user','content':'Revenue?'},{'role':'assistant','content':'$4900'}]}
    (source/'review-candidates.jsonl').write_text(json.dumps(row)+'\n')
    decision={'id':'fixture','status':'approved','required_prompt_text':['$4900'],
              'observations_sha256':{'observation.json':hashlib.sha256(path.read_bytes()).hexdigest()}}
    reviews=tmp_path/'decisions.jsonl';reviews.write_text(json.dumps(decision)+'\n')
    return source,reviews,path,row,decision


def test_review_rejects_changed_evidence_before_writing(tmp_path):
    source,reviews,path,_,_=fixture(tmp_path)
    path.write_text(path.read_text().replace('$4900','$490'))
    with pytest.raises(ValueError,match='observation changed'):
        prepare(source,reviews,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_review_does_not_move_validation_into_training(tmp_path):
    source,reviews,_,row,_=fixture(tmp_path)
    row['split']='validation'
    (source/'review-candidates.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='held-out groups'):
        prepare(source,reviews,tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_review_rejects_missing_answer_evidence(tmp_path):
    source,reviews,_,_,decision=fixture(tmp_path)
    decision['required_prompt_text']=['$13000']
    reviews.write_text(json.dumps(decision)+'\n')
    with pytest.raises(ValueError,match='grounding was omitted'):
        prepare(source,reviews,tmp_path/'output')
    assert not (tmp_path/'output').exists()
