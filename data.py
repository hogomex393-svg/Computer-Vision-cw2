import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from runtime import letterbox, normalize, overlay

LABELS = ['background', 'transparent_thing', 'transparent_stuff']


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


def audit(root, out, encoding, official=True):
    root, out = Path(root).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    errors, duplicate_groups, all_hashes, splits = [], [], {}, {}
    values, pixel_counts, dimensions = {}, np.zeros(4, dtype=np.int64), {}
    for split in ('train', 'validation', 'test'):
        split_root = root / split
        records = []
        image_dirs = sorted(p for p in split_root.rglob('images') if p.is_dir())
        if not image_dirs:
            errors.append('Missing images directory: ' + split)
        for directory in image_dirs:
            mask_dir = directory.parent / 'masks'
            used_masks = set()
            for image_path in sorted(directory.iterdir()):
                if not image_path.is_file():
                    continue
                candidates = [mask_dir / (image_path.stem + '_mask.png'), mask_dir / (image_path.stem + '.png')]
                found = [p for p in candidates if p.is_file()]
                if len(found) != 1:
                    errors.append('Missing/ambiguous mask: ' + str(image_path.relative_to(root)))
                    continue
                mask_path = found[0]
                used_masks.add(mask_path)
                try:
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
                        for v in unique:
                            key = str(v)
                            values[key] = values.get(key, 0) + 1
                        mask = decode_mask_image(raw, encoding, str(mask_path))
                    if rgb.shape[:2] != mask.shape:
                        raise ValueError('Image/mask dimensions differ')
                    for i, c in enumerate((0, 1, 2, 255)):
                        pixel_counts[i] += (mask == c).sum()
                    dim = '{}x{}'.format(rgb.shape[1], rgb.shape[0])
                    dimensions[dim] = dimensions.get(dim, 0) + 1
                    digest = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
                    location = str(image_path.relative_to(root)).replace('\\', '/')
                    all_hashes.setdefault(digest, []).append({'split': split, 'image': location})
                    record = {'image': location, 'mask': str(mask_path.relative_to(root)).replace('\\', '/'),
                              'height': rgb.shape[0], 'width': rgb.shape[1], 'decoded_sha256': digest,
                              'group': location.rsplit('/', 2)[0]}
                    records.append(record)
                    if len(records) % 100 == 0:
                        (out / 'audit_progress.json').write_text(json.dumps({'split': split, 'processed': len(records), 'errors': len(errors)}), encoding='utf-8')
                    if len(records) <= 2:
                        preview = np.concatenate([rgb, overlay(rgb, mask)], axis=1)
                        Image.fromarray(preview).save(out / ('audit_{}_{}.png'.format(split, len(records))))
                except (ValueError, OSError) as error:
                    errors.append('{}: {}'.format(image_path, error))
            if mask_dir.exists():
                for unused in set(p for p in mask_dir.iterdir() if p.is_file()) - used_masks:
                    errors.append('Unpaired mask: ' + str(unused.relative_to(root)))
        splits[split] = records
    for digest, group in all_hashes.items():
        if len(group) > 1:
            duplicate_groups.append({'sha256': digest, 'files': group})
            if len({r['split'] for r in group}) > 1:
                errors.append('Cross-split duplicate: ' + digest)
    counts = {k: len(v) for k, v in splits.items()}
    if official and counts != {'train': 5000, 'validation': 1000, 'test': 4428}:
        errors.append('Official split counts differ: ' + str(counts))
    report = {'schema': 1, 'source': 'Trans10K v1' if official else 'synthetic-smoke', 'encoding': encoding,
              'counts': counts, 'raw_values_image_counts': values, 'class_pixels': dict(zip(LABELS + ['ignore'], pixel_counts.tolist())),
              'dimensions': dimensions, 'duplicates': duplicate_groups, 'errors': errors, 'passed': not errors}
    (out / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for split, records in splits.items():
        manifest = {'schema': 1, 'source': report['source'], 'encoding': encoding, 'split': split,
                    'audit_passed': not errors, 'records': records}
        (out / (split + '.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if errors:
        raise ValueError('Dataset audit failed; inspect ' + str(out / 'audit.json'))
    return report


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
        if cache:
            cache = Path(cache)
            record = json.loads((cache / 'cache.json').read_text())
            if record['manifest_sha256'] != sha256(manifest) or record['size'] != size:
                raise ValueError('Cache does not match manifest/size')
            self.cache_images = np.load(cache / 'images.npy', mmap_mode='r')
            self.cache_masks = np.load(cache / 'masks.npy', mmap_mode='r')
            self.cache_meta = record['geometry']

    def __len__(self):
        return len(self.records)

    def original(self, index):
        record = self.records[index]
        with Image.open(self.root / record['image']) as im:
            rgb = np.asarray(im.convert('RGB')).copy()
        return rgb, load_mask(self.root / record['mask'], self.manifest['encoding'])

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


def prepare_cache(root, manifest, out, size):
    out = Path(out)
    if (out / 'cache.json').exists():
        record = json.loads((out / 'cache.json').read_text())
        if record['manifest_sha256'] != sha256(manifest) or record['size'] != size:
            raise ValueError('Existing cache differs; choose a new cache directory')
        return record
    out.mkdir(parents=True, exist_ok=True)
    data = SegmentationData(root, manifest, size)
    images = np.lib.format.open_memmap(out / 'images.npy', mode='w+', dtype=np.uint8, shape=(len(data), size, size, 3))
    masks = np.lib.format.open_memmap(out / 'masks.npy', mode='w+', dtype=np.uint8, shape=(len(data), size, size))
    geometry = []
    for i in range(len(data)):
        image, mask = data.original(i)
        images[i], masks[i], meta = letterbox(image, size, mask)
        geometry.append(meta)
    images.flush()
    masks.flush()
    record = {'manifest_sha256': sha256(manifest), 'size': size, 'geometry': geometry,
              'source': data.manifest['source'], 'samples': len(data)}
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
