import argparse
import json
from pathlib import Path
import shutil
import zipfile

import torch

from data import sha256
from learning import load_checkpoint, write_json


def package(runs, output):
    runs, output = Path(runs), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Artifact staging must be new/empty; do not overwrite an existing release')
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((runs / 'selection.json').read_text())
    selected = selection['selected']
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        run = json.loads((runs / name / 'run.json').read_text())
        if run['status'] != 'completed' or run['smoke']:
            raise ValueError('A formal experiment is incomplete: ' + name)
        checkpoint = load_checkpoint(runs / name / 'best.pt')
        light = {key: checkpoint[key] for key in ('config', 'model', 'smoke', 'epoch', 'identity', 'validation', 'validation_manifest_sha256')}
        light['original_checkpoint_sha256'] = sha256(runs / name / 'best.pt')
        destination = output / 'checkpoints' / (name + '.pt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(light, destination)
        log_dir = output / 'results' / name
        log_dir.mkdir(parents=True, exist_ok=True)
        for filename in ('run.json', 'epochs.jsonl', 'best_validation.json', 'test_clean.json', 'test_brightness.json', 'test_blur.json', 'test_jpeg.json'):
            shutil.copy2(runs / name / filename, log_dir / filename)
    models = output / 'models'
    models.mkdir()
    for size in (256, 384, 512):
        for suffix in ('.onnx', '.json', '.reference.npz'):
            filename = '{}_{}{}'.format(selected, size, suffix)
            shutil.copy2(runs / 'exports' / filename, models / filename)
    shutil.copy2(runs / 'selection.json', output / 'selection.json')
    for filename in ('nano.py', 'runtime.py'):
        shutil.copy2(Path(__file__).parent / filename, output / filename)
    (output / 'ARTIFACTS.md').write_text(
        '# Trained artifacts\n\nSelected by validation: ' + selected +
        '. Inference checkpoints omit optimizer/RNG state; full resume checkpoints remain in the training run.\n'
        'No TensorRT engine is included: build on the target Nano. Nano hardware acceptance is separate.\n'
        'Use the matching source Git revision and docs/JETSON.md. Public-data accuracy is in results; all measurements retain provenance.\n', encoding='utf-8')
    files = [{'path': str(path.relative_to(output)).replace('\\', '/'), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
             for path in sorted(output.rglob('*')) if path.is_file()]
    write_json(output / 'manifest.json', {'selected': selected, 'files': files})
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Release archive exists; select a new version')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(output)))
    record = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha256(archive),
              'selected': selected, 'download_url': None, 'access': 'pending private group storage',
              'nano_hardware_verified': False}
    write_json(archive.with_suffix('.release.json'), record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.runs, args.out), indent=2))
