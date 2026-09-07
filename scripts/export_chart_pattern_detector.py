#!/usr/bin/env python3
"""Restricted-load the local YOLO checkpoint, export ONNX, and verify parity."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'outputs/chart-agent/yolo-config'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'outputs/chart-agent/matplotlib'))

# Explicitly reviewed constructors present in the supplied checkpoint. Unknown
# pickle globals are rejected; never switch to weights_only=False on failure.
ALLOWED = '''torch.nn.modules.activation.SiLU
torch.nn.modules.batchnorm.BatchNorm2d
torch.nn.modules.container.ModuleList
torch.nn.modules.container.Sequential
torch.nn.modules.conv.Conv2d
torch.nn.modules.loss.BCEWithLogitsLoss
torch.nn.modules.pooling.MaxPool2d
torch.nn.modules.upsampling.Upsample
ultralytics.nn.modules.block.Bottleneck
ultralytics.nn.modules.block.C2f
ultralytics.nn.modules.block.DFL
ultralytics.nn.modules.block.SPPF
ultralytics.nn.modules.conv.Concat
ultralytics.nn.modules.conv.Conv
ultralytics.nn.modules.head.Detect
ultralytics.nn.tasks.DetectionModel
ultralytics.utils.IterableSimpleNamespace
ultralytics.utils.loss.BboxLoss
ultralytics.utils.loss.v8DetectionLoss
ultralytics.utils.tal.TaskAlignedAssigner'''.splitlines()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'outputs/chart-agent/assets/chart-pattern.onnx')
    args = p.parse_args()
    import torch
    import numpy as np
    import onnx
    import onnxruntime as ort

    unknown = set(torch.serialization.get_unsafe_globals_in_checkpoint(args.checkpoint)) - set(ALLOWED)
    if unknown:
        raise ValueError('Unreviewed checkpoint globals: ' + ', '.join(sorted(unknown)))
    constructors = [getattr(importlib.import_module(name.rsplit('.', 1)[0]), name.rsplit('.', 1)[1]) for name in ALLOWED]
    with torch.serialization.safe_globals(constructors):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.head import Detect
    model = checkpoint['model']
    if not isinstance(model, DetectionModel):
        raise ValueError('Expected a YOLO DetectionModel')
    model = model.float().eval()
    torch.set_num_threads(2)
    for module in model.modules():
        if isinstance(module, Detect):
            module.export, module.format, module.dynamic = True, 'onnx', False
    size = int(checkpoint.get('train_args', {}).get('imgsz', 640))
    if not 32 <= size <= 2048 or size % 32:
        raise ValueError('Unsupported checkpoint image size')
    torch.manual_seed(42)
    sample = torch.rand(1, 3, size, size)
    with torch.inference_mode():
        reference = model(sample).numpy()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, sample, str(args.output), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'])
    graph = onnx.load(args.output)
    onnx.helper.set_model_props(graph, {'names': repr(model.names), 'imgsz': repr([size, size]),
        'task': 'detect', 'description': 'ChartScanAI local Buy/Sell label detector; accuracy not established',
        'source_sha256': hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest()})
    onnx.checker.check_model(graph)
    onnx.save(graph, args.output)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(args.output), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'images': sample.numpy()})[0]
    np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
    report = dict(names=model.names, input_shape=list(sample.shape), output_shape=list(actual.shape),
                  max_absolute_difference=float(np.max(np.abs(actual - reference))), numerical_parity=True,
                  checkpoint_sha256=hashlib.file_digest(args.checkpoint.open('rb'), 'sha256').hexdigest(),
                  onnx_sha256=hashlib.file_digest(args.output.open('rb'), 'sha256').hexdigest(),
                  torch_version=torch.__version__, heldout_accuracy_evaluated=False)
    args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
