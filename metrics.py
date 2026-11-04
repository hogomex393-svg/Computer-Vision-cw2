import cv2
import numpy as np


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def edges(labels, valid, category):
    binary = labels == category
    edge = np.zeros(labels.shape, bool)
    h = (binary[:, 1:] != binary[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    v = (binary[1:] != binary[:-1]) & valid[1:] & valid[:-1]
    edge[:, 1:] |= h
    edge[:, :-1] |= h
    edge[1:] |= v
    edge[:-1] |= v
    return edge


class Metrics:
    def __init__(self, tolerance=2):
        self.confusion = np.zeros((3, 3), dtype=np.int64)
        self.tolerance = tolerance
        self.boundary = np.zeros(4, dtype=np.int64)

    def update(self, prediction, target):
        if prediction.shape != target.shape:
            raise ValueError('Metric shapes differ')
        valid = target != 255
        if not np.isin(target[valid], [0, 1, 2]).all() or not np.isin(prediction[valid], [0, 1, 2]).all():
            raise ValueError('Invalid metric class')
        counts = np.bincount((target[valid] * 3 + prediction[valid]).astype(int), minlength=9).reshape(3, 3)
        self.confusion += counts
        kernel = np.ones((2 * self.tolerance + 1, 2 * self.tolerance + 1), np.uint8)
        safe = cv2.erode(valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
        for category in (1, 2):
            pred = edges(prediction, valid, category) & safe
            true = edges(target, valid, category) & safe
            matched_pred = pred & (cv2.dilate(true.astype(np.uint8), kernel) > 0)
            matched_true = true & (cv2.dilate(pred.astype(np.uint8), kernel) > 0)
            self.boundary += np.array([matched_pred.sum(), pred.sum(), matched_true.sum(), true.sum()])
        return counts

    def result(self):
        matrix = self.confusion
        intersection = np.diag(matrix)
        union = matrix.sum(0) + matrix.sum(1) - intersection
        iou = [ratio(a, b) for a, b in zip(intersection, union)]
        present = [x for x in iou if x is not None]
        tp = matrix[1:, 1:].sum()
        fp, fn = matrix[0, 1:].sum(), matrix[1:, 0].sum()
        precision, recall = ratio(*self.boundary[:2]), ratio(*self.boundary[2:])
        if self.boundary[1] == 0 and self.boundary[3] == 0:
            bf1 = None
        elif precision is None or recall is None or precision + recall == 0:
            bf1 = 0.0
        else:
            bf1 = 2 * precision * recall / (precision + recall)
        return {'miou': float(np.mean(present)) if present else None, 'per_class_iou': iou,
                'foreground_iou': ratio(tp, tp + fp + fn), 'foreground_dice': ratio(2 * tp, 2 * tp + fp + fn),
                'boundary_f1': bf1, 'boundary_tolerance_pixels': self.tolerance,
                'confusion': matrix.tolist(), 'valid_pixels': int(matrix.sum()),
                'absent_class_policy': 'exclude only classes with zero union',
                'boundary_policy': 'micro over foreground classes; Chebyshev tolerance at original resolution; empty/empty=null'}


def miou_from_confusion(counts):
    intersection = np.diag(counts)
    union = counts.sum(0) + counts.sum(1) - intersection
    valid = union > 0
    return float(np.mean(intersection[valid] / union[valid])) if valid.any() else None


def tensor_statistics(prediction, target, tolerance=2):
    import torch
    from torch.nn import functional as F
    if prediction.shape != target.shape:
        raise ValueError('Metric shapes differ')
    valid = target != 255
    truths = [target == c for c in (0, 1, 2)]
    predictions = [prediction == c for c in (0, 1, 2)]
    if (valid & ~(truths[0] | truths[1] | truths[2])).any() or (valid & ~(predictions[0] | predictions[1] | predictions[2])).any():
        raise ValueError('Invalid metric class')
    counts = torch.stack([(a & b & valid).sum() for a in truths for b in predictions]).reshape(3, 3)
    def dilate(binary):
        return F.max_pool2d(binary[None, None].float(), 2 * tolerance + 1, 1, tolerance)[0, 0] > 0
    safe = valid if valid.all() else valid & ~dilate(~valid)
    def boundary(binary):
        result = torch.zeros_like(valid)
        horizontal = (binary[:, 1:] != binary[:, :-1]) & valid[:, 1:] & valid[:, :-1]
        vertical = (binary[1:] != binary[:-1]) & valid[1:] & valid[:-1]
        result[:, 1:] |= horizontal
        result[:, :-1] |= horizontal
        result[1:] |= vertical
        result[:-1] |= vertical
        return result & safe
    boundary_counts = torch.zeros(4, dtype=torch.int64, device=prediction.device)
    for category in (1, 2):
        pred = boundary(predictions[category])
        true = boundary(truths[category])
        boundary_counts += torch.stack([(pred & dilate(true)).sum(), pred.sum(),
                                        (true & dilate(pred)).sum(), true.sum()])
    return counts.cpu().numpy(), boundary_counts.cpu().numpy()


def corrupt(rgb, kind, seed=42):
    rng = np.random.default_rng(seed)
    if kind == 'clean':
        return rgb.copy()
    if kind == 'brightness':
        return np.clip(rgb.astype(float) * 0.6, 0, 255).astype(np.uint8)
    if kind == 'blur':
        return cv2.GaussianBlur(rgb, (7, 7), 1.5)
    if kind == 'jpeg':
        quality = int(rng.integers(30, 31))
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    raise ValueError('Unknown corruption: ' + kind)
