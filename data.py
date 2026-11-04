import hashlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from filelock import FileLock

from runtime import letterbox, normalize, overlay, preprocess
from metrics import corrupt

LABELS = ['background', 'transparent_thing', 'transparent_stuff']


def cache_protocol():
    return hashlib.sha256((inspect.getsource(letterbox) + inspect.getsource(corrupt)).encode()).hexdigest()


def load_mask(path, encoding):
    with Image.open(path) as image:
        return decode_mask_image(image, encoding, str(path))


def decode_mask_image(image, encoding, name='mask'):
    if encoding == 'trans10k_v1':
        a = np.asarray(image.convert('RGB'), dtype=np.uint16).sum(-1, dtype=np.uint16)
        mapping, length = {0: 0, 255: 1, 765: 2}, 766
    elif encoding == 'indexed':
        a = np.asarray(image)
        if a.ndim != 2 or not np.issubdtype(a.dtype, np.integer):
            raise ValueError('Indexed labels must be single-channel integers')
        if a.min() < 0 or a.max() > 255:
            raise ValueError('Unknown indexed mask values')
        mapping, length = {0: 0, 1: 1, 2: 2, 255: 255}, 256
    else:
        raise ValueError('Explicit mask encoding required')
    lookup = np.full(length, -1, np.int16)
    for raw, target in mapping.items():
        lookup[raw] = target
    result = lookup[a]
    unknown = result < 0
    if unknown.any():
        raise ValueError('Unknown mask values in {}: {}'.format(name, np.unique(a[unknown]).tolist()))
    return result.astype(np.uint8)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_sample(job):
    image_path, mask_path, location, mask_location, encoding, correction, signature, preview = job
    result = {'image': location, 'signature': signature}
    try:
        if correction:
            if not correction.get('reviewed') or correction['mask_rotation_cw'] not in (0, 90, 180, 270):
                raise ValueError('Geometry correction is not reviewed/valid')
            if sha256(image_path) != correction['image_sha256'] or sha256(mask_path) != correction['mask_sha256']:
                raise ValueError('Geometry correction SHA256 mismatch')
        with Image.open(image_path) as image:
            image.load()
            rgb = np.asarray(image.convert('RGB'))
        with Image.open(mask_path) as raw:
            raw.load()
            colored = raw.convert('RGB') if encoding == 'trans10k_v1' else raw
            colors = colored.getcolors(1024)
            if colors is not None:
                unique = [list(value) if isinstance(value, tuple) else value for _, value in colors]
            else:
                arr = np.asarray(colored)
                unique = np.unique(arr.reshape(-1, 3), axis=0).tolist() if arr.ndim == 3 else np.unique(arr).tolist()
            mask = decode_mask_image(raw, encoding, mask_location)
        rotation = correction.get('mask_rotation_cw', 0)
        raw_shape = list(mask.shape)
        if rotation:
            mask = np.rot90(mask, -(rotation // 90)).copy()
        if rgb.shape[:2] != mask.shape:
            raise ValueError('Image/mask dimensions differ: {} vs {}'.format(rgb.shape[:2], mask.shape))
        h = hashlib.sha256()
        h.update(rgb.data)
        h.update(str(rgb.shape).encode())
        record = {'image': location, 'mask': mask_location, 'height': rgb.shape[0], 'width': rgb.shape[1],
                  'decoded_sha256': h.hexdigest(), 'group': location.rsplit('/', 2)[0],
                  'raw_mask_shape': raw_shape, 'mask_rotation_cw': rotation}
        result.update(record=record, colors=unique,
                      pixels=np.bincount(mask.ravel(), minlength=256)[[0, 1, 2, 255]].tolist())
        if preview:
            figure = Image.fromarray(np.concatenate([rgb, overlay(rgb, mask)], axis=1))
            figure.thumbnail((1200, 700))
            figure.save(preview)
    except (ValueError, OSError, Image.DecompressionBombError) as error:
        result['error'] = '{}: {}'.format(location, error)
    return result


def audit(root, out, encoding, official=True, geometry_overrides=None, workers=4):
    Path(out).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(out) / '.audit.lock'), timeout=0):
        return _audit(root, out, encoding, official, geometry_overrides, workers)


def _audit(root, out, encoding, official=True, geometry_overrides=None, workers=4):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    corrections = json.loads(Path(geometry_overrides).read_text())['corrections'] if geometry_overrides else {}
    journal_path = out / 'audit_rows.jsonl'
    previous = {}
    if journal_path.exists():
        lines = journal_path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                previous[row['image']] = row
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError('Audit journal is corrupted before its final line')
                journal_path.write_text('\n'.join(lines[:index]) + '\n', encoding='utf-8')
        if journal_path.stat().st_size:
            with journal_path.open('rb') as handle:
                handle.seek(-1, 2)
                needs_newline = handle.read(1) != b'\n'
            if needs_newline:
                with journal_path.open('ab') as handle:
                    handle.write(b'\n')
    protocol = hashlib.sha256((inspect.getsource(decode_mask_image) + inspect.getsource(audit_sample)).encode()).hexdigest()
    errors, duplicate_groups, all_hashes, splits = [], [], {}, {}
    values, pixel_counts, dimensions = {}, np.zeros(4, dtype=np.int64), {}
    reused = 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool, journal_path.open('a', encoding='utf-8') as journal:
        for split in ('train', 'validation', 'test'):
            jobs, records = [], []
            image_dirs = sorted(p for p in (root / split).rglob('images') if p.is_dir())
            if not image_dirs:
                errors.append('Missing images directory: ' + split)
            for directory in image_dirs:
                mask_dir, used_masks = directory.parent / 'masks', set()
                for image_path in sorted(directory.iterdir()):
                    if not image_path.is_file():
                        continue
                    found = [p for p in (mask_dir / (image_path.stem + '_mask.png'), mask_dir / (image_path.stem + '.png')) if p.is_file()]
                    if len(found) != 1:
                        errors.append('Missing/ambiguous mask: ' + str(image_path.relative_to(root)))
                        continue
                    mask_path = found[0]
                    used_masks.add(mask_path)
                    location = image_path.relative_to(root).as_posix()
                    correction = corrections.get(location, {})
                    signature = [image_path.stat().st_size, image_path.stat().st_mtime_ns,
                                 mask_path.stat().st_size, mask_path.stat().st_mtime_ns, encoding, correction, protocol]
                    preview = out / ('audit_{}_{}.png'.format(split, len(jobs) + 1)) if len(jobs) < 2 else None
                    jobs.append((image_path, mask_path, location, mask_path.relative_to(root).as_posix(), encoding, correction, signature, preview))
                if mask_dir.exists():
                    for unused in set(p for p in mask_dir.iterdir() if p.is_file()) - used_masks:
                        errors.append('Unpaired mask: ' + str(unused.relative_to(root)))
            def checked(job):
                cached = previous.get(job[2])
                if cached and cached['signature'] == job[6]:
                    return dict(cached, reused=True)
                return audit_sample(job)
            for index, row in enumerate(pool.map(checked, jobs)):
                if row.get('reused'):
                    reused += 1
                else:
                    journal.write(json.dumps(row) + '\n')
                if 'error' in row:
                    errors.append(row['error'])
                else:
                    record = row['record']
                    records.append(record)
                    pixel_counts += np.asarray(row['pixels'], np.int64)
                    for value in row['colors']:
                        key = str(value)
                        values[key] = values.get(key, 0) + 1
                    dim = '{}x{}'.format(record['width'], record['height'])
                    dimensions[dim] = dimensions.get(dim, 0) + 1
                    all_hashes.setdefault(record['decoded_sha256'], []).append({'split': split, 'image': record['image']})
                if (index + 1) % 100 == 0:
                    journal.flush()
                    temporary = out / 'audit_progress.tmp'
                    temporary.write_text(json.dumps({'split': split, 'processed': index + 1, 'errors': errors, 'reused': reused}), encoding='utf-8')
                    temporary.replace(out / 'audit_progress.json')
            splits[split] = records
    for digest, group in all_hashes.items():
        if len(group) > 1:
            duplicate_groups.append({'sha256': digest, 'files': group})
            if len({r['split'] for r in group}) > 1:
                errors.append('Cross-split duplicate: ' + digest)
    counts = {k: len(v) for k, v in splits.items()}
    if official and counts != {'train': 5000, 'validation': 1000, 'test': 4428}:
        errors.append('Official split counts differ: ' + str(counts))
    report = {'schema': 2, 'source': 'Trans10K v1' if official else 'synthetic-smoke', 'encoding': encoding,
              'counts': counts, 'raw_values_image_counts': values, 'class_pixels': dict(zip(LABELS + ['ignore'], pixel_counts.tolist())),
              'dimensions': dimensions, 'duplicates': duplicate_groups, 'errors': errors, 'passed': not errors,
              'geometry_corrections': corrections, 'journal_reused': reused, 'audit_protocol_sha256': protocol}
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for split, records in splits.items():
        manifest = {'schema': 1, 'source': report['source'], 'encoding': encoding, 'split': split,
                    'audit_passed': not errors, 'records': records}
        (out / (split + '.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if errors:
        raise ValueError('Dataset audit failed; inspect ' + str(out / 'audit.json'))
    return report


def prepare_splits(audit_directory, out):
    audit_directory, out = Path(audit_directory), Path(out)
    report = json.loads((audit_directory / 'audit.json').read_text(encoding='utf-8'))
    manifests = {split: json.loads((audit_directory / (split + '.json')).read_text(encoding='utf-8'))
                 for split in ('train', 'validation', 'test')}
    groups = {}
    for split, manifest in manifests.items():
        for row in manifest['records']:
            groups.setdefault(row['decoded_sha256'], set()).add(split)
    expected_errors = {'Cross-split duplicate: ' + digest for digest, splits in groups.items() if len(splits) > 1}
    if set(report['errors']) != expected_errors:
        raise ValueError('Cannot prepare derived splits while non-duplicate audit failures remain')
    original_counts = {split: len(manifest['records']) for split, manifest in manifests.items()}
    if original_counts != report['counts']:
        raise ValueError('Audit/manifest count mismatch')
    if report['source'] == 'Trans10K v1' and original_counts != {'train': 5000, 'validation': 1000, 'test': 4428}:
        raise ValueError('Official source split counts differ')
    test_hashes = {r['decoded_sha256'] for r in manifests['test']['records']}
    validation = [r for r in manifests['validation']['records'] if r['decoded_sha256'] not in test_hashes]
    holdout = test_hashes | {r['decoded_sha256'] for r in validation}
    training = [r for r in manifests['train']['records'] if r['decoded_sha256'] not in holdout]
    active = {'train': training, 'validation': validation, 'test': manifests['test']['records']}
    active_hashes = {split: {r['decoded_sha256'] for r in records} for split, records in active.items()}
    if any(active_hashes[a] & active_hashes[b] for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]):
        raise AssertionError('Derived split leakage remains')
    excluded = {split: [r['image'] for r in manifests[split]['records'] if r['image'] not in {a['image'] for a in active[split]}]
                for split in active}
    counts = {split: len(records) for split, records in active.items()}
    record = {'schema': 1, 'passed': True, 'source': report['source'], 'protocol': 'leak-free-v1',
              'source_audit_sha256': sha256(audit_directory / 'audit.json'), 'original_counts': original_counts,
              'counts': counts, 'excluded': excluded, 'test_preserved': True,
              'policy': 'test retained; remove test duplicates from validation; remove holdout duplicates from train',
              'source_duplicate_groups': report['duplicates'], 'cross_split_duplicates': [],
              'geometry_corrections': report.get('geometry_corrections', {})}
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in active.items():
        result = dict(manifests[split], records=rows, audit_passed=True, protocol='leak-free-v1',
                      original_counts=original_counts, effective_counts=counts, excluded=excluded[split],
                      source_audit_sha256=record['source_audit_sha256'])
        (out / (split + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    (out / 'audit.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


class SegmentationData(Dataset):
    def __init__(self, root, manifest, size, augment=False, seed=42, mean=None, std=None, cache=None):
        self.root = Path(root)
        self.manifest = json.loads(Path(manifest).read_text(encoding='utf-8'))
        if not self.manifest['audit_passed']:
            raise ValueError('Only a passed audit can be used for training/evaluation')
        self.records = self.manifest['records']
        self.size, self.augment, self.seed, self.epoch = size, augment, seed, 0
        self.mean, self.std = mean, std
        self.cache_images = self.cache_masks = self.cache_meta = None
        self.cache_inputs = {}
        self.cache_seed = None
        if cache:
            cache = Path(cache)
            record = json.loads((cache / 'cache.json').read_text())
            if record['manifest_sha256'] != sha256(manifest) or record['size'] != size:
                raise ValueError('Cache does not match manifest/size')
            if record.get('protocol_sha256') != cache_protocol():
                raise ValueError('Cache preprocessing code differs; rebuild the cache')
            self.cache_images = np.load(cache / 'images.npy', mmap_mode='r')
            self.cache_masks = np.load(cache / 'masks.npy', mmap_mode='r')
            self.cache_meta = record['geometry']
            self.cache_seed = record.get('seed', 42)
            self.cache_inputs['clean'] = self.cache_images
            for condition in record.get('conditions', ['clean']):
                if condition != 'clean':
                    self.cache_inputs[condition] = np.load(cache / (condition + '.npy'), mmap_mode='r')

    def __len__(self):
        return len(self.records)

    def original(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        mask = self.target(index)
        if rgb.shape[:2] != mask.shape:
            raise ValueError('Audited image/mask geometry changed')
        return rgb, mask

    def target(self, index):
        record = self.records[index]
        mask = load_mask(self.root / record['mask'], self.manifest['encoding'])
        rotation = record.get('mask_rotation_cw', 0)
        if rotation:
            mask = np.rot90(mask, -(rotation // 90)).copy()
        if (record['height'], record['width']) != mask.shape:
            raise ValueError('Audited image/mask geometry changed')
        return mask

    def evaluation_item(self, index, condition='clean'):
        kwargs = {'mean': self.mean, 'std': self.std} if self.mean is not None else {}
        if condition in self.cache_inputs:
            if condition != 'clean' and self.cache_seed != self.seed:
                raise ValueError('Corruption cache seed differs from evaluation')
            tensor = normalize(self.cache_inputs[condition][index], **kwargs)[None]
            return tensor, self.cache_meta[index], self.target(index)
        rgb, target = self.original(index)
        image = corrupt(rgb, condition, self.seed + index)
        tensor, meta = preprocess(image, self.size, **kwargs)
        return tensor, meta, target

    def __getitem__(self, index):
        if self.cache_images is not None:
            rgb, mask, meta = self.cache_images[index].copy(), self.cache_masks[index].copy(), self.cache_meta[index]
        else:
            rgb, mask = self.original(index)
            rgb, mask, meta = letterbox(rgb, self.size, mask)
        top, left, h, w = meta['top'], meta['left'], meta['resized_height'], meta['resized_width']
        rgb, mask = rgb[top:top + h, left:left + w], mask[top:top + h, left:left + w]
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        if self.augment:
            if rng.random() < 0.5:
                rgb, mask = rgb[:, ::-1], mask[:, ::-1]
            rgb = np.clip(rgb.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8)
        rgb, mask, _ = letterbox(rgb, self.size, mask)
        kwargs = {'mean': self.mean, 'std': self.std} if self.mean is not None else {}
        return torch.from_numpy(normalize(rgb, **kwargs)), torch.from_numpy(mask.astype(np.int64))


def prepare_cache(root, manifest, out, size, conditions=None, seed=42, workers=4):
    Path(out).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(out) / '.cache.lock'), timeout=0):
        return _prepare_cache(root, manifest, out, size, conditions, seed, workers)


def _prepare_cache(root, manifest, out, size, conditions=None, seed=42, workers=4):
    out = Path(out)
    conditions = list(dict.fromkeys(['clean'] + (conditions or [])))
    if (out / 'cache.json').exists():
        record = json.loads((out / 'cache.json').read_text())
        if record['manifest_sha256'] != sha256(manifest) or record['size'] != size:
            raise ValueError('Existing cache differs; choose a new cache directory')
        if record.get('conditions', ['clean']) != conditions or record.get('seed', 42) != seed:
            raise ValueError('Existing cache appearance protocol differs')
        if record.get('protocol_sha256') != cache_protocol():
            raise ValueError('Existing cache preprocessing code differs')
        return {k: record[k] for k in ('samples', 'size', 'manifest_sha256')}
    out.mkdir(parents=True, exist_ok=True)
    data = SegmentationData(root, manifest, size)
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(data), size, size, 3))
    masks = np.lib.format.open_memmap(out / 'masks.npy', mode='w+', dtype=np.uint8, shape=(len(data), size, size))
    variants = {condition: np.lib.format.open_memmap(out / (condition + '.npy'), mode='w+', dtype=np.uint8,
                                                    shape=(len(data), size, size, 3)) for condition in conditions if condition != 'clean'}
    geometry = []
    def prepare(i):
        image, mask = data.original(i)
        clean, target, meta = letterbox(image, size, mask)
        changed = {condition: letterbox(corrupt(image, condition, seed + i), size)[0] for condition in variants}
        return clean, target, meta, changed
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        for start in range(0, len(data), 16):
            for i, result in enumerate(pool.map(prepare, range(start, min(start + 16, len(data)))), start):
                clean, target, meta, changed = result
                images[i], masks[i] = clean, target
                geometry.append(meta)
                for condition, array in changed.items():
                    variants[condition][i] = array
            temporary = out / 'cache_progress.tmp'
            temporary.write_text(json.dumps({'processed': len(geometry), 'total': len(data)}))
            temporary.replace(out / 'cache_progress.json')
    images.flush()
    masks.flush()
    for variant in variants.values():
        variant.flush()
    record = {'manifest_sha256': sha256(manifest), 'size': size, 'geometry': geometry,
              'source': data.manifest['source'], 'samples': len(data), 'conditions': conditions, 'seed': seed,
              'protocol_sha256': cache_protocol()}
    (out / 'cache.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return {'samples': len(data), 'size': size, 'manifest_sha256': record['manifest_sha256']}


def synthetic(root, count=12, seed=42):
    root = Path(root)
    for split, n in [('train', count), ('validation', 4), ('test', 4)]:
        rng = np.random.default_rng(seed + {'train': 0, 'validation': 100, 'test': 200}[split])
        for sub in ('images', 'masks'):
            (root / split / sub).mkdir(parents=True, exist_ok=True)
        for index in range(n):
            h, w = int(rng.integers(64, 90)), int(rng.integers(80, 115))
            mask = np.zeros((h, w), np.uint8)
            x = int(rng.integers(4, 16))
            cv2.rectangle(mask, (x, 8), (x + 22, h - 12), 1, -1)
            cv2.circle(mask, (w - 24, h // 2), 16, 2, -1)
            rgb = np.clip(np.array([[25, 35, 45], [35, 220, 65], [220, 55, 45]])[mask] +
                          rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(rgb).save(root / split / 'images' / ('smoke_{:03}.png'.format(index)))
            Image.fromarray(mask).save(root / split / 'masks' / ('smoke_{:03}_mask.png'.format(index)))
    (root / 'SYNTHETIC_ONLY.txt').write_text('Synthetic colored shapes, NOT transparent-object benchmark data.\n')
