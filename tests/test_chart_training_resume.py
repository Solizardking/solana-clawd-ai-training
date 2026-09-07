import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('chart_training', Path(__file__).parents[1] / 'scripts/train_chart_foundation.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def checkpoint(tmp_path):
    root = tmp_path / 'prior'
    path = root / 'sft/checkpoint-100'
    path.mkdir(parents=True)
    for name in ('adapter_config.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
        (path / name).write_text('fixture')
    (path / 'trainer_state.json').write_text(json.dumps({'global_step': 100}))
    manifest = {'counts': {'train': 30}, 'source_sha256': 'original'}
    (root / 'data-manifest.json').write_text(json.dumps(manifest))
    return path, manifest


def test_resume_accepts_complete_matching_checkpoint(tmp_path):
    path, manifest = checkpoint(tmp_path)
    assert training.validate_resume(path, 'sft', manifest) == path.resolve()


def test_resume_rejects_changed_dataset(tmp_path):
    path, manifest = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='manifest differs'):
        training.validate_resume(path, 'sft', {**manifest, 'source_sha256': 'changed'})


def test_resume_requires_optimizer_state_and_correct_stage(tmp_path):
    path, manifest = checkpoint(tmp_path)
    with pytest.raises(ValueError, match='selected stage'):
        training.validate_resume(path, 'documents', manifest)
    (path / 'optimizer.pt').unlink()
    with pytest.raises(ValueError, match='optimizer.pt'):
        training.validate_resume(path, 'sft', manifest)
