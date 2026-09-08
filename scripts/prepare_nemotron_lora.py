#!/usr/bin/env python3
"""Prepare a local NeMo BF16 LoRA pilot; never allocate GPUs or upload artifacts."""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISION = 'a9904d24bcc1d289a1950fa9d2b978c47cf903b9'
SOURCE_URL = 'https://raw.githubusercontent.com/NVIDIA-NeMo/Nemotron/main/usage-cookbook/Nemotron-3.5-Lightning/dgx-station-recipes/lora_lightning35_station.yaml'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(data, output, token_audit=None, full_epoch=False):
    data, output = data.resolve(), output.resolve()
    if output.exists():
        raise ValueError('Use a new output directory; existing runs are preserved')
    manifest = json.loads((data / 'manifest.json').read_text())
    audit_path = token_audit or ROOT / 'outputs/nemotron-token-length-audit.json'
    audit = json.loads(audit_path.read_text())
    if audit.get('valid') is False:
        raise ValueError('Token audit is marked invalid')
    counts = {}
    hashes = {}
    for split in ('train', 'validation'):
        path = data / f'{split}.jsonl'
        hashes[path.name] = digest(path)
        if hashes[path.name] != manifest['outputs_sha256'][path.name]:
            raise ValueError(f'{split} dataset hash differs from the preparation manifest')
        if hashes[path.name] != audit['splits'][split]['input_sha256']:
            raise ValueError(f'{split} token audit belongs to different data')
        counts[split] = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('images') or not isinstance(row.get('messages'), list):
                    raise ValueError('Pilot requires prepared text-only agent-chat rows')
                if not any(m.get('role') == 'assistant' and m.get('content') for m in row['messages']):
                    raise ValueError('Missing supervised assistant content')
                if not isinstance(row.get('tools'), str) or not isinstance(json.loads(row['tools']), list):
                    raise ValueError('Agent-chat tools must be a serialized JSON list')
                counts[split] += 1
        if not counts[split]:
            raise ValueError(f'{split} dataset is empty')
    upstream = ROOT / 'deploy/nemotron/training/upstream-lora.yaml'
    config = yaml.safe_load(upstream.read_text())
    mask_template = ROOT / 'deploy/nemotron/training/assistant-mask.jinja'
    mask_text = mask_template.read_text()
    maximum = max(audit['splits'][split]['max_tokens'] for split in ('train','validation'))
    sequence_length = max(4096, 1 << (maximum - 1).bit_length())
    if sequence_length > 32768:
        raise ValueError('Audited data exceeds the upstream 32768 context recipe')
    config['model']['revision'] = MODEL_REVISION
    # Verified on A100 with the longest 13,035-token training example.
    config['model']['output_hidden_states'] = True
    config['loss_fn'] = {
        '_target_': 'nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy'
    }
    config.setdefault('distributed', {})['activation_checkpointing'] = True
    config['step_scheduler'].update(local_batch_size=1, global_batch_size=8,
                                    max_steps=100, ckpt_every_steps=25, val_every_steps=25)
    if full_epoch:
        config['step_scheduler'].update(num_epochs=1, max_steps=None,
                                        ckpt_every_steps=1000, val_every_steps=1000)
    config['checkpoint']['checkpoint_dir'] = '/shared/run/checkpoints'
    for key, split in [('dataset', 'train'), ('validation_dataset', 'validation')]:
        config[key]['path'] = f'/shared/data/{split}.jsonl'
        config[key]['seq_length'] = sequence_length
        config[key]['tokenizer']['revision'] = MODEL_REVISION
        config[key]['tokenizer']['chat_template'] = mask_text
    output.mkdir(parents=True)
    (output / 'cache').mkdir()
    (output / 'pilot.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    command = ['docker', 'run', '--rm', '--gpus', 'all', '--ipc=host',
               '--mount', f'type=bind,src={data},dst=/shared/data,readonly',
               '--mount', f'type=bind,src={output},dst=/shared/run',
               '-e', 'HF_HOME=/shared/run/cache', '-e', 'HF_TOKEN',
               '-w', '/opt/Automodel', 'nvcr.io/nvidia/nemo-automodel:26.08',
               'automodel', '/shared/run/pilot.yaml', '--nproc-per-node', '1']
    (output / 'run-pilot.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n' + shlex.join(command) + '\n')
    report = {'purpose': 'One epoch over the supplied prepared training set' if full_epoch else '100-step pilot, not complete dataset training',
              'requested_epochs': 1 if full_epoch else None,
              'expected_optimizer_steps': (counts['train'] + 7) // 8 if full_epoch else 100,
              'source_url': SOURCE_URL, 'source_sha256': digest(upstream),
              'dataset_sha256': hashes, 'counts': counts,
              'config_sha256': digest(output / 'pilot.yaml'),
              'gpu_executed': False, 'training_completed': False,
              'framework_ingestion_verified': False,
              'visual_pending': manifest.get('counts', {}).get('train', {}).get('visual_pending'),
              'model_revision': MODEL_REVISION,
              'assistant_mask_template_sha256': digest(mask_template),
              'sequence_length': sequence_length,
              'token_audit_sha256': digest(audit_path),
              'limitations': ['Native NeMo formatting and loss-mask retention remain unverified',
                  'Visual examples and document pretraining are separate unfinished stages',
                  'GPU memory, loss, throughput and checkpoint reload must be measured']}
    (output / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'outputs/nemotron-chart-data')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/nemotron-lora-pilot')
    parser.add_argument('--token-audit', type=Path, default=ROOT / 'outputs/nemotron-token-length-audit.json')
    parser.add_argument('--full-epoch', action='store_true', help='Prepare one full epoch instead of the 100-step pilot; does not launch training')
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.output, args.token_audit, args.full_epoch), indent=2))
