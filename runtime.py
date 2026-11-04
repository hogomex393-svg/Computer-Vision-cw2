import json
import os
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
COLORS = np.array([[0, 0, 0], [30, 210, 100], [245, 130, 35]], dtype=np.uint8)


def read_image(path):
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def write_image(path, image):
    ok, encoded = cv2.imencode(Path(path).suffix, image)
    if not ok:
        raise IOError('Cannot encode image: ' + str(path))
    encoded.tofile(str(path))


def video_path(path):
    return os.path.relpath(str(Path(path).resolve()))


def letterbox(rgb, size, mask=None):
    height, width = rgb.shape[:2]
    ratio = min(float(size) / height, float(size) / width)
    h, w = max(1, int(round(height * ratio))), max(1, int(round(width * ratio)))
    top, left = (size - h) // 2, (size - w) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[top:top + h, left:left + w] = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    meta = {'height': height, 'width': width, 'top': top, 'left': left,
            'resized_height': h, 'resized_width': w, 'size': size}
    target = None
    if mask is not None:
        target = np.full((size, size), 255, dtype=np.uint8)
        target[top:top + h, left:left + w] = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return canvas, target, meta


def normalize(rgb, mean=MEAN, std=STD):
    values = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(((values - np.asarray(mean, np.float32)) /
                                np.asarray(std, np.float32)).transpose(2, 0, 1))


def preprocess(rgb, size, mean=MEAN, std=STD):
    canvas, _, meta = letterbox(rgb, size)
    return normalize(canvas, mean, std)[None], meta


def restore_logits(logits, meta):
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError('Runtime requires batch=1')
        logits = logits[0]
    values = np.asarray(logits, np.float32).transpose(1, 2, 0)
    size = meta['size']
    if values.shape[:2] != (size, size):
        values = cv2.resize(values, (size, size), interpolation=cv2.INTER_LINEAR)
    top, left = meta['top'], meta['left']
    values = values[top:top + meta['resized_height'], left:left + meta['resized_width']]
    return cv2.resize(values, (meta['width'], meta['height']), interpolation=cv2.INTER_LINEAR)


def overlay(rgb, mask, alpha=0.45):
    result = rgb.copy()
    foreground = (mask > 0) & (mask < 3)
    result[foreground] = ((1 - alpha) * rgb[foreground] + alpha * COLORS[mask[foreground]]).astype(np.uint8)
    return result


def latency_summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {'count': 0, 'mean_ms': None, 'p50_ms': None, 'p95_ms': None, 'fps': None}
    return {'count': len(a), 'mean_ms': float(a.mean()), 'p50_ms': float(np.percentile(a, 50)),
            'p95_ms': float(np.percentile(a, 95)), 'fps': float(1000 / a.mean()) if a.mean() else None}


class ONNXRunner:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0]
        self.size = self.input.shape[-1]
        metadata = json.loads(Path(path).with_suffix('.json').read_text())
        self.mean, self.std = metadata['normalization']['mean'], metadata['normalization']['std']
        if self.input.shape != [1, 3, self.size, self.size] or not isinstance(self.size, int):
            raise ValueError('Expected fixed [1,3,H,H] model')

    def __call__(self, tensor):
        start = time.perf_counter()
        result = self.session.run(None, {self.input.name: tensor})[0]
        return result, (time.perf_counter() - start) * 1000

    def close(self):
        self.session = None


class LatestFrames:
    def __init__(self, capture):
        self.capture = capture
        self.queue = queue.Queue(maxsize=1)
        self.stop = threading.Event()
        self.done = threading.Event()
        self.dropped = 0
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.stop.is_set():
                ok, frame = self.capture.read()
                if not ok:
                    break
                item = (frame, time.perf_counter())
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                        self.dropped += 1
                    except queue.Empty:
                        pass
                    self.queue.put_nowait(item)
        finally:
            self.done.set()

    def read(self):
        while True:
            try:
                return self.queue.get(timeout=0.1)
            except queue.Empty:
                if self.done.is_set():
                    return None

    def close(self):
        self.stop.set()
        self.capture.release()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Capture thread failed to stop')


def demo(runner, source, out, headless=True, asynchronous=False, warmup=5,
         duration=0, max_frames=0, label='model', smoke=False, loop=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inference_ms, end_to_end_ms, age_ms = [], [], []
    image_types = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    path = Path(str(source))
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in image_types) if path.is_dir() else (
        [path] if path.suffix.lower() in image_types else [])
    capture = latest = writer = None
    status = 'SYNTHETIC SMOKE - NOT A REAL MODEL' if smoke else label
    start_total = time.perf_counter()
    count = 0
    if path.is_dir() and not images:
        raise ValueError('No images in source directory')
    try:
        for _ in range(warmup):
            runner(np.zeros((1, 3, runner.size, runner.size), np.float32))
        if not images:
            camera = str(source).isdigit()
            capture = cv2.VideoCapture(int(source) if camera else video_path(source))
            if not capture.isOpened():
                raise RuntimeError('Cannot open camera/video: ' + str(source))
            if asynchronous and loop:
                raise ValueError('Async capture and playback loop cannot be combined')
            latest = LatestFrames(capture) if asynchronous else None
        start_total = time.perf_counter()
        while True:
            start = time.perf_counter()
            if images:
                if count >= len(images):
                    break
                bgr = read_image(images[count])
                if bgr is None:
                    raise ValueError('Unreadable image: ' + str(images[count]))
                captured = start
            elif latest:
                item = latest.read()
                if item is None:
                    break
                bgr, captured = item
            else:
                ok, bgr = capture.read()
                if not ok and loop:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = capture.read()
                if not ok:
                    break
                captured = time.perf_counter()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor, meta = preprocess(rgb, runner.size, runner.mean, runner.std)
            logits, elapsed = runner(tensor)
            mask = restore_logits(logits, meta).argmax(-1).astype(np.uint8)
            panel = np.concatenate([rgb, overlay(rgb, mask)], axis=1)
            panel = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            display_width = min(1280, max(640, panel.shape[1]))
            if display_width != panel.shape[1]:
                display_height = max(1, int(round(panel.shape[0] * float(display_width) / panel.shape[1])))
                panel = cv2.resize(panel, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
            panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            cv2.putText(panel, '{} | {} | {:.1f} ms'.format(status, runner.size, elapsed),
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if images:
                name = images[count].stem
                write_image(out / (name + '_overlay.png'), panel)
                write_image(out / (name + '_mask.png'), mask)
            else:
                if writer is None:
                    fps = capture.get(cv2.CAP_PROP_FPS)
                    fps = fps if np.isfinite(fps) and fps > 0 else 25
                    writer = cv2.VideoWriter(video_path(out / 'demo.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                                             fps, (panel.shape[1], panel.shape[0]))
                    if not writer.isOpened():
                        raise IOError('Cannot initialize video writer')
                writer.write(panel)
            if not headless:
                cv2.imshow('Transparent segmentation', panel)
                if cv2.waitKey(1) & 0xff == 27:
                    break
            end = time.perf_counter()
            inference_ms.append(elapsed)
            end_to_end_ms.append((end - start) * 1000)
            age_ms.append((end - captured) * 1000)
            count += 1
            if (max_frames and count >= max_frames) or (duration and end - start_total >= duration):
                break
        if count == 0:
            raise RuntimeError('No frames processed')
    finally:
        if latest:
            latest.close()
        elif capture:
            capture.release()
        if writer:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()
    record = {'label': status, 'smoke': smoke, 'input_size': runner.size,
              'source_type': 'images' if images else ('live' if str(source).isdigit() else 'playback'),
              'async': asynchronous, 'frames': count, 'dropped': latest.dropped if latest else 0,
              'inference': latency_summary(inference_ms), 'end_to_end': latency_summary(end_to_end_ms),
              'frame_age': latency_summary(age_ms), 'wall_seconds': time.perf_counter() - start_total}
    record['processed_fps_wall'] = count / record['wall_seconds']
    (out / 'timings.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record
