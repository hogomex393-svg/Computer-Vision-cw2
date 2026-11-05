import argparse
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import torch

from data import sha256
from learning import load_checkpoint, write_json


def package(runs, output, manifests=None, demo_dir=None):
    runs, output = Path(runs), Path(output)
    pipeline_state = runs / 'pipeline.json'
    if not pipeline_state.exists() or json.loads(pipeline_state.read_text())['status'] != 'completed':
        raise ValueError('The complete training/evaluation/export pipeline must finish before packaging')
    if set(json.loads(pipeline_state.read_text())['experiments']) != {'T', 'S0', 'S1', 'S2', 'S3'}:
        raise ValueError('All five experiments must be represented in the completed pipeline')
    if manifests is None:
        raise ValueError('Pass the audited experiment manifests')
    if demo_dir is None:
        raise ValueError('Pass the verified validation replay directory')
    manifests = Path(manifests)
    manifest_hashes = {split: sha256(manifests / (split + '.json')) for split in ('train', 'validation', 'test')}
    dirty = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', '*.py', 'configs', 'requirements*.txt'])
    if dirty.returncode:
        raise ValueError('Commit the project code before creating a source-bound artifact package')
    source_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], check=True, capture_output=True, text=True).stdout.strip()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Artifact staging must be new/empty; do not overwrite an existing release')
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((runs / 'selection.json').read_text())
    selected = selection['selected']
    if selected not in ('S0', 'S1', 'S2', 'S3'):
        raise ValueError('Deployment selection must be a student')
    demo_dir = Path(demo_dir)
    demo_record = json.loads((demo_dir / 'provenance.json').read_text())
    if demo_record['selected'] != selected or demo_record['source_type'] != 'validation_still_image_replay':
        raise ValueError('Demo provenance differs from the selected model')
    (output / 'demo').mkdir()
    for name in ('input_replay.mp4', 'S0.mp4', 'final.mp4', 'S0_timings.json', 'final_timings.json'):
        if sha256(demo_dir / name) != demo_record['files'][name]:
            raise ValueError('Demo file differs: ' + name)
        shutil.copy2(demo_dir / name, output / 'demo' / name)
    shutil.copy2(demo_dir / 'provenance.json', output / 'demo/provenance.json')
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        run = json.loads((runs / name / 'run.json').read_text())
        if run['status'] != 'completed' or run['smoke']:
            raise ValueError('A formal experiment is incomplete: ' + name)
        for split in ('train', 'validation'):
            if run['provenance']['manifests'][split + '.json'] != manifest_hashes[split]:
                raise ValueError('Dataset manifest differs from the trained experiment')
        checkpoint = load_checkpoint(runs / name / 'best.pt')
        if checkpoint['identity'] != run['identity'] or checkpoint['config'] != run['config'] or checkpoint['smoke']:
            raise ValueError('Checkpoint/run provenance mismatch: ' + name)
        light = {key: checkpoint[key] for key in ('config', 'model', 'smoke', 'epoch', 'global_step', 'identity', 'validation', 'validation_manifest_sha256')}
        light['original_checkpoint_sha256'] = sha256(runs / name / 'best.pt')
        destination = output / 'checkpoints' / (name + '.pt')
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(light, destination)
        log_dir = output / 'results' / name
        log_dir.mkdir(parents=True, exist_ok=True)
        for filename in ('run.json', 'epochs.jsonl', 'best_validation.json', 'test_clean.json', 'test_brightness.json', 'test_blur.json', 'test_jpeg.json'):
            if filename.startswith('test_') and json.loads((runs / name / filename).read_text())['manifest_sha256'] != manifest_hashes['test']:
                raise ValueError('Test manifest provenance mismatch')
            shutil.copy2(runs / name / filename, log_dir / filename)
        if name == 'T' and (runs / name / 'teacher_reuse.json').exists():
            shutil.copy2(runs / name / 'teacher_reuse.json', log_dir / 'teacher_reuse.json')
    (output / 'manifests').mkdir()
    for filename in ('train.json', 'validation.json', 'test.json', 'audit.json'):
        shutil.copy2(manifests / filename, output / 'manifests' / filename)
    models = output / 'models'
    models.mkdir()
    export_pairs = {(selected, size) for size in (256, 384, 512)} | {('S0', 384)}
    for model_name, size in sorted(export_pairs):
        for suffix in ('.onnx', '.json', '.reference.npz'):
            filename = '{}_{}{}'.format(model_name, size, suffix)
            shutil.copy2(runs / 'exports' / filename, models / filename)
    shutil.copy2(runs / 'selection.json', output / 'selection.json')
    for filename in ('nano.py', 'runtime.py'):
        shutil.copy2(Path(__file__).parent / filename, output / filename)
    (output / 'ARTIFACTS.md').write_text(
        '# Trained artifacts\n\nSelected by validation: ' + selected +
        '. Source commit: ' + source_commit +
        '. Inference checkpoints omit optimizer/RNG state; full resume checkpoints remain in the training run.\n'
        'No TensorRT engine is included: build on the target Nano. Nano hardware acceptance is separate.\n'
        'Classes: 0=background, 1=transparent things, 2=transparent stuff. Dataset ignore=255 is not a model output class.\n'
        'The S0 384 baseline export is included for sequential comparison on the same video.\n'
        'demo/ contains a validation still-image replay and both model outputs. It is not live footage or a Nano speed test.\n'
        'Use the matching source Git revision and docs/JETSON.md. Public-data accuracy is in results; all measurements retain provenance.\n\n'
        '```bash\npython3 nano.py env --out nano_results/environment\n'
        'python3 nano.py build --onnx models/' + selected + '_384.onnx --engine models/' + selected + '_384_fp16.engine --precision fp16\n'
        'python3 nano.py verify --engine models/' + selected + '_384_fp16.engine --reference models/' + selected + '_384.reference.npz --out nano_results/alignment\n'
        'python3 nano.py demo --engine models/' + selected + '_384_fp16.engine --source demo/input_replay.mp4 --out nano_results/playback\n```\n', encoding='utf-8')
    files = [{'path': str(path.relative_to(output)).replace('\\', '/'), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
             for path in sorted(output.rglob('*')) if path.is_file()]
    write_json(output / 'manifest.json', {'selected': selected, 'source_commit': source_commit, 'files': files})
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Release archive exists; select a new version')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(output)))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None or set(z.namelist()) != {f['path'] for f in files} | {'manifest.json'}:
            raise ValueError('Artifact archive content verification failed')
    record = {'archive': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha256(archive),
              'selected': selected, 'source_commit': source_commit, 'download_url': None,
              'access': 'pending collaborator-only draft release',
              'nano_hardware_verified': False}
    write_json(archive.with_suffix('.release.json'), record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--manifests', required=True)
    parser.add_argument('--demo-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(package(args.runs, args.out, args.manifests, args.demo_dir), indent=2))
