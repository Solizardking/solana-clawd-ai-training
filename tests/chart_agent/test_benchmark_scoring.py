import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('visual_eval', Path(__file__).resolve().parents[2] / 'scripts/eval_chart_visual_benchmark.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_quantities_and_units():
    assert module.answer_matches('52', '52 units')
    assert module.answer_matches('52', '52.0')
    assert not module.answer_matches('52', '152')
    assert not module.answer_matches('52', '52%')
    assert not module.answer_matches('52', '52 or 53')


def test_exact_labels():
    assert module.answer_matches('Cedar', 'cedar.')
    assert not module.answer_matches('Cedar', 'Cedar or Elm')
    assert not module.answer_matches('Cedar', 'Not Cedar')
