import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('served_eval', Path(__file__).resolve().parents[2]/'scripts/evaluate_served_chart_detector.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ServedDetectorScoringTests(unittest.TestCase):
    def test_duplicate_prediction_cannot_match_same_object_twice(self):
        predictions = [dict(confidence=c, bbox_xyxy=[0, 0, 10, 10]) for c in [0.7, 0.9]]
        self.assertEqual(module.match(predictions, [[0, 0, 10, 10]]), (1, 1, 0))

    def test_missed_and_spurious_objects_are_counted(self):
        predictions = [dict(confidence=0.9, bbox_xyxy=[20, 20, 30, 30])]
        self.assertEqual(module.match(predictions, [[0, 0, 10, 10]]), (0, 1, 1))

    def test_empty_predictions_preserve_false_negatives(self):
        self.assertEqual(module.match([], [[0, 0, 10, 10]]), (0, 0, 1))


if __name__ == '__main__':
    unittest.main()
