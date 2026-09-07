import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('chartdete_prepare', Path(__file__).parents[1] / 'scripts/prepare_chartdete.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_box_conversion_clips_to_image():
    assert module.normalized_box([-10, 20, 50, 100], 100, 100) == (0.2, 0.6, 0.4, 0.8)
    assert module.normalized_box([110, 20, 10, 10], 100, 100) is None


def test_nonfinite_and_degenerate_boxes():
    with pytest.raises(ValueError):
        module.normalized_box([float('nan'), 0, 10, 10], 100, 100)
    assert module.normalized_box([0, 0, -1, 10], 100, 100) is None
