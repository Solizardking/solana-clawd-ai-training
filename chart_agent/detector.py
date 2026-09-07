"""YOLOv8 ONNX detection with letterboxing and class-aware NMS.

Only numeric ONNX graphs are loaded; no pickle-based torch checkpoints.
Class labels must come from the graph metadata, never an assumed COCO list.
"""
import ast
import io
import numpy as np
from PIL import Image, ImageOps


def decode_image(data):
    image = Image.open(io.BytesIO(data))
    if image.width * image.height > 16_000_000:
        raise ValueError('Image exceeds 16 million pixels')
    if image.format not in {'PNG', 'JPEG', 'WEBP'}:
        raise ValueError('Use PNG, JPEG, or WebP')
    return ImageOps.exif_transpose(image).convert('RGB')


def nms(boxes, scores, threshold=0.45):
    order, kept = np.argsort(-scores), []
    while order.size:
        i = int(order[0])
        kept.append(i)
        rest = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        intersection = np.prod(np.maximum(hi - lo, 0), axis=1)
        area = np.prod(np.maximum(boxes[:, 2:] - boxes[:, :2], 0), axis=1)
        iou = intersection / np.maximum(area[i] + area[rest] - intersection, 1e-9)
        order = rest[iou <= threshold]
    return kept


class Detector:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        self.names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map['names'])
        if isinstance(self.names, list):
            self.names = dict(enumerate(self.names))
        self.names = {int(k): str(v) for k, v in self.names.items()}
        shape = self.input.shape
        if len(shape) != 4 or shape[1] != 3 or not all(isinstance(v, int) for v in shape[2:]):
            raise ValueError('Expected static NCHW RGB YOLO input')
        self.height, self.width = shape[2:]

    def detect(self, image, confidence=0.35):
        w, h = image.size
        scale = min(self.width / w, self.height / h)
        rw, rh = round(w * scale), round(h * scale)
        left, top = (self.width - rw) // 2, (self.height - rh) // 2
        canvas = Image.new('RGB', (self.width, self.height), (114, 114, 114))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
        tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
        pred = self.session.run(None, {self.input.name: tensor})[0]
        if pred.ndim != 3 or pred.shape[0] != 1 or pred.shape[1] != 4 + len(self.names):
            raise ValueError('Unsupported YOLO output; expected [1, 4+classes, anchors]')
        pred = pred[0].T
        classes = pred[:, 4:].argmax(axis=1)
        scores = pred[:, 4:].max(axis=1)
        mask = np.isfinite(pred).all(axis=1) & (scores >= confidence)
        pred, classes, scores = pred[mask], classes[mask], scores[mask]
        boxes = np.concatenate([pred[:, :2] - pred[:, 2:4] / 2, pred[:, :2] + pred[:, 2:4] / 2], axis=1)
        boxes = (boxes - np.array([left, top, left, top])) / scale
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        result = []
        for cls in np.unique(classes):
            indices = np.where(classes == cls)[0]
            for j in nms(boxes[indices], scores[indices]):
                i = indices[j]
                result.append(dict(label=self.names[int(cls)], confidence=float(scores[i]), bbox_xyxy=boxes[i].round(2).tolist()))
        return sorted(result, key=lambda x: -x['confidence'])[:100]
