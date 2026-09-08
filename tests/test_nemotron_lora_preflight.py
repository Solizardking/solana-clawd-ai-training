import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from scripts.prepare_nemotron_lora import MODEL_REVISION, prepare


class PilotPreflightTests(unittest.TestCase):
    def test_pin_and_reject_changed_dataset_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.mkdir()
            hashes = {}
            for split in ('train', 'validation'):
                path = data / f'{split}.jsonl'
                path.write_text(json.dumps({'messages': [{'role': 'user', 'content': 'Convert'},
                    {'role': 'assistant', 'content': '123.456789'}], 'tools': '[]'}) + '\n')
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (data / 'manifest.json').write_text(json.dumps({'outputs_sha256': hashes}))
            audit = root / 'audit.json'
            audit.write_text(json.dumps({'splits': {split: {'input_sha256': hashes[f'{split}.jsonl'], 'max_tokens': 13035} for split in ('train', 'validation')}}))
            report = prepare(data, root / 'run', audit)
            config = yaml.safe_load((root / 'run/pilot.yaml').read_text())
            self.assertFalse(report['gpu_executed'])
            self.assertEqual(config['model']['revision'], MODEL_REVISION)
            self.assertEqual(config['dataset']['tokenizer']['revision'], MODEL_REVISION)
            self.assertEqual(config['step_scheduler']['max_steps'], 100)
            self.assertEqual(config['dataset']['seq_length'], 16384)
            self.assertTrue(config['model']['output_hidden_states'])
            self.assertTrue(config['distributed']['activation_checkpointing'])
            self.assertEqual(config['loss_fn']['_target_'],
                'nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy')
            self.assertEqual(config['peft']['exclude_modules'], ['*.out_proj'])
            full = prepare(data, root / 'full', audit, full_epoch=True)
            full_config = yaml.safe_load((root / 'full/pilot.yaml').read_text())
            self.assertEqual(full_config['step_scheduler']['num_epochs'], 1)
            self.assertIsNone(full_config['step_scheduler']['max_steps'])
            self.assertEqual(full['expected_optimizer_steps'], 1)
            with self.assertRaisesRegex(ValueError, 'existing runs'):
                prepare(data, root / 'run', audit)
            with (data / 'train.jsonl').open('a') as stream:
                stream.write('\n')
            with self.assertRaisesRegex(ValueError, 'hash differs'):
                prepare(data, root / 'changed', audit)
            self.assertFalse((root / 'changed').exists())


if __name__ == '__main__':
    unittest.main()
