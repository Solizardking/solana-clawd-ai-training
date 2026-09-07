#!/usr/bin/env python3
"""Check converted LoRA tensor shapes against the exact served GGUF."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from gguf import GGUFReader


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--adapter', type=Path, required=True)
    args = p.parse_args()
    base, adapter = GGUFReader(args.base), GGUFReader(args.adapter)
    base_tensors = {t.name: t for t in base.tensors}
    tensors = {t.name: t for t in adapter.tensors}
    verified = []
    for name, a in tensors.items():
        if not name.endswith('.lora_a'):
            continue
        weight = name.removesuffix('.lora_a')
        b = tensors[weight + '.lora_b']
        target = base_tensors[weight]
        if len(target.shape) != 2 or list(a.shape) != [target.shape[0], b.shape[0]] or b.shape[1] != target.shape[1]:
            raise ValueError('LoRA/base tensor shape mismatch: ' + weight)
        if not np.isfinite(a.data).all() or not np.isfinite(b.data).all():
            raise ValueError('Nonfinite adapter weights: ' + weight)
        verified.append(weight)
    if not verified or len(verified) * 2 != len(tensors):
        raise ValueError('Missing or unpaired LoRA tensors')
    with args.adapter.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {'adapter_sha256': digest, 'base_file': args.base.name, 'paired_base_weights': len(verified),
              'all_finite': True, 'tensor_shapes_compatible': True,
              'runtime_application_verified': False, 'accuracy_improvement_verified': False}
    args.adapter.with_suffix('.verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
