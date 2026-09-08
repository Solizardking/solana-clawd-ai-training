"""Exercise completion handoff without training, network calls, or real artifacts."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "training_watch", Path(__file__).resolve().parents[2] / "scripts/watch_chart_training.py"
)
watch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watch)


class TrainingWatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "full").mkdir()
        (self.root / "full/results.csv").write_text("epoch,loss\n3,0.8\n")
        for name, value in (("RUN", self.root), ("STATE", self.root / "state.json")):
            mock = patch.object(watch, name, value)
            mock.start()
            self.addCleanup(mock.stop)
        api = patch.object(watch, "HfApi")
        self.api = api.start()
        self.addCleanup(api.stop)
        self.api.return_value.inspect_job.return_value = SimpleNamespace(
            id="test-job", status=SimpleNamespace(stage="RUNNING")
        )
        self.record = {"job_id": "test-job", "job_url": "https://example.invalid/job"}

    def complete(self):
        (self.root / "full-result.json").write_text(json.dumps({
            "training_finished": True, "smoke_only": False
        }))

    def tick(self, process_active=False, evaluation_code=0):
        calls = []

        def run(command, **kwargs):
            if command[0] == "ps":
                return SimpleNamespace(returncode=0 if process_active else 1,
                    stdout="python scripts/train_chartdete_detector.py" if process_active else "")
            calls.append(command)
            if evaluation_code == 0:
                (self.root / "full/evaluation/report.json").write_text("{}")
            return SimpleNamespace(returncode=evaluation_code)

        with patch.object(watch.subprocess, "run", side_effect=run), patch("builtins.print"):
            state = watch.tick(123, self.record, True)
        return state, calls

    def test_completion_record_does_not_override_running_process(self):
        self.complete()
        state, calls = self.tick(process_active=True)
        self.assertTrue(state["detector"]["process_active"])
        self.assertEqual(calls, [])

    def test_missing_completion_requires_review_without_evaluation(self):
        state, calls = self.tick()
        self.assertTrue(state["detector"]["needs_recovery_review"])
        self.assertEqual(calls, [])

    def test_success_runs_once_after_exit(self):
        self.complete()
        state, calls = self.tick()
        self.assertEqual(len(calls), 1)
        self.assertTrue(state["detector"]["evaluation_report_available"])
        self.assertEqual(self.tick()[1], [])

    def test_failed_evaluation_is_not_retried(self):
        self.complete()
        state, calls = self.tick(evaluation_code=1)
        self.assertEqual(len(calls), 1)
        self.assertFalse(state["detector"]["evaluation_report_available"])
        attempt = json.loads((self.root / "full/evaluation/monitor-attempt.json").read_text())
        self.assertEqual(attempt["status"], "failed_needs_review")
        self.assertEqual(self.tick()[1], [])

    def test_hub_observation_failure_does_not_infer_terminal_state(self):
        self.api.return_value.inspect_job.side_effect = TimeoutError()
        state, _ = self.tick(process_active=True)
        self.assertTrue(state["llm"]["terminal_state_not_inferred"])
        self.assertNotIn("stage", state["llm"])


if __name__ == "__main__":
    unittest.main()
