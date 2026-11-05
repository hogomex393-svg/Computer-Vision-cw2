import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time


def config_file(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def evaluation_is_current(path, identity, checkpoint_sha, manifest_sha, condition, sample_count):
    if not Path(path).exists():
        return False
    try:
        result = config_file(path)
        return (result.get('run_identity') == identity and result.get('checkpoint_sha256') == checkpoint_sha and
                result.get('manifest_sha256') == manifest_sha and result.get('corruption') == condition and
                result.get('samples') == sample_count and len(result.get('per_image', [])) == sample_count and
                result.get('smoke') is False)
    except (ValueError, OSError):
        return False


def export_is_current(path, checkpoint_sha, size):
    from data import sha256
    path = Path(path)
    metadata_path, reference = path.with_suffix('.json'), path.with_suffix('.reference.npz')
    if not all(p.is_file() for p in (path, metadata_path, reference)):
        return False
    try:
        metadata = config_file(metadata_path)
        return (metadata.get('checkpoint_sha256') == checkpoint_sha and metadata.get('shape') == [1, 3, size, size] and
                metadata.get('opset') == 11 and metadata.get('smoke') is False and
                metadata.get('auxiliary_removed_exact') is True and
                metadata.get('onnx_sha256') == sha256(path) and metadata.get('reference_sha256') == sha256(reference))
    except (ValueError, OSError):
        return False


def prepare_figure_manifest(manifest_path, evaluation_path, out, checkpoint_sha, limit=6):
    from data import sha256
    from learning import write_json
    manifest, evaluation = config_file(manifest_path), config_file(evaluation_path)
    if (evaluation['manifest_sha256'] != sha256(manifest_path) or
            evaluation['checkpoint_sha256'] != checkpoint_sha or evaluation['corruption'] != 'clean' or evaluation['smoke']):
        raise ValueError('Figure selection requires the matching complete clean-test result')
    scores = {row['image']: row['miou'] for row in evaluation['per_image']}
    records = manifest['records']
    if set(scores) != {row['image'] for row in records} or evaluation['samples'] != len(records):
        raise ValueError('Figure selection cannot use partial test results')
    ranked = sorted((scores[row['image']] if scores[row['image']] is not None else 1, i) for i, row in enumerate(records))
    indices = list(range(min(limit, len(records))))
    for _, i in ranked[:limit]:
        if i not in indices:
            indices.append(i)
    selected = dict(manifest, source='Trans10K v1 figure subset', split='figures_only', protocol='figure-selection-only',
                    records=[records[i] for i in indices], full_test_indices=indices,
                    source_manifest_sha256=sha256(manifest_path), source_evaluation_sha256=sha256(evaluation_path))
    write_json(out, selected)
    return indices


def render_comparison(runs, selected, root, manifest, device):
    from data import sha256
    from learning import comparison, write_json
    runs = Path(runs)
    out = runs / 'comparison'
    out.mkdir(parents=True, exist_ok=True)
    expected = {'baseline_checkpoint_sha256': sha256(runs / 'S0/best.pt'),
                'final_checkpoint_sha256': sha256(runs / selected / 'best.pt'),
                'manifest_sha256': sha256(manifest), 'evaluation_sha256': sha256(runs / selected / 'test_clean.json')}
    receipt_path = out / 'comparison_receipt.json'
    if receipt_path.exists():
        receipt = config_file(receipt_path)
        if all(receipt.get(key) == value for key, value in expected.items()) and all(
                (out / entry['path']).is_file() and sha256(out / entry['path']) == entry['sha256'] for entry in receipt.get('files', [])) and receipt.get('files'):
            return receipt
    figure_manifest = out / 'figure_manifest.json'
    indices = prepare_figure_manifest(manifest, runs / selected / 'test_clean.json', figure_manifest, expected['final_checkpoint_sha256'])
    comparison(runs / 'S0/best.pt', runs / selected / 'best.pt', root, figure_manifest, out, device)
    selection = config_file(out / 'selection.json')
    selection.update(full_test_indices=indices, fixed_global_indices=[indices[i] for i in selection['fixed_indices']],
                     failure_global_indices=[indices[i] for i in selection['failure_indices']],
                     final_model=selected, source_manifest_sha256=expected['manifest_sha256'],
                     failure_rule='lowest per-image mIoU from the complete clean test, tie by original manifest index')
    write_json(out / 'selection.json', selection)
    names = ['selection.json', 'figure_manifest.json']
    names += ['fixed_{:03}.png'.format(i) for i in selection['fixed_indices']]
    names += ['failure_{:03}.png'.format(i) for i in selection['failure_indices']]
    receipt = dict(expected, files=[{'path': name, 'sha256': sha256(out / name)} for name in names])
    write_json(receipt_path, receipt)
    return receipt


def environment(out):
    import importlib.metadata as metadata
    import torch
    from learning import write_json
    packages = ['torch', 'torchvision', 'numpy', 'Pillow', 'opencv-python', 'transformers', 'onnx', 'onnxruntime', 'pytest']
    record = {'os': platform.platform(), 'python': sys.version, 'executable': sys.executable,
              'packages': {p: metadata.version(p) for p in packages}, 'cuda_available': torch.cuda.is_available(),
              'cuda_build': torch.version.cuda, 'disk_free_bytes': shutil.disk_usage(Path.cwd()).free,
              'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
              'nano': 'unavailable: no target connection provided'}
    write_json(out, record)
    return record


def reuse_teacher(source, destination, manifests, expected_config):
    from data import sha256
    from learning import load_checkpoint, write_json
    source, destination, manifests = Path(source), Path(destination), Path(manifests)
    if source.resolve() == destination.resolve():
        raise ValueError('Teacher source and destination must be different')
    run = config_file(source / 'run.json')
    if (run['status'] != 'completed' or run['smoke'] or run['config'] != expected_config or
            run['config']['kind'] != 'teacher' or run['last_completed_epoch'] != expected_config['epochs']):
        raise ValueError('Only a complete, matching formal teacher can be reused')
    digest = hashlib.sha256(json.dumps(run['provenance'], sort_keys=True).encode()).hexdigest()
    if digest != run['identity'] or run['provenance']['config'] != expected_config:
        raise ValueError('Teacher provenance identity differs')
    for split in ('train', 'validation'):
        if run['provenance']['manifests'][split + '.json'] != sha256(manifests / (split + '.json')):
            raise ValueError('Teacher data manifests differ')
    for name in ('best', 'last'):
        checkpoint = load_checkpoint(source / (name + '.pt'))
        if (checkpoint['identity'] != run['identity'] or checkpoint['config'] != expected_config or checkpoint['smoke'] or
                checkpoint['validation_manifest_sha256'] != run['provenance']['manifests']['validation.json'] or
                (name == 'last' and checkpoint['epoch'] != expected_config['epochs']) or
                (name == 'best' and checkpoint['validation'] != config_file(source / 'best_validation.json'))):
            raise ValueError('Teacher checkpoint provenance differs')
        del checkpoint
    names = ('run.json', 'epochs.jsonl', 'best_validation.json', 'best.pt', 'last.pt')
    hashes = {name: sha256(source / name) for name in names}
    receipt = {'run_identity': run['identity'], 'original_code_sources': run['provenance']['sources'],
               'original_commit': run['git_commit_at_start'], 'files': hashes,
               'rule': 'Reuse a completed teacher without changing its original config, identity, checkpoints or training records'}
    existing = destination / 'teacher_reuse.json'
    if destination.exists() and any(destination.iterdir()):
        if not existing.exists() or config_file(existing) != receipt:
            raise ValueError('Destination is not the same verified teacher import')
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(source / name, destination / name)
        write_json(existing, receipt)
    if any(sha256(destination / name) != value for name, value in hashes.items()):
        raise ValueError('Copied teacher files differ')
    return run


def pipeline(args):
    from learning import write_json, record_progress
    try:
        result = run_pipeline(args)
        write_json(Path(args.out) / 'pipeline.json', {'status': 'completed', 'experiments': args.experiments})
        stage = ('主机训练、测试、导出流水线完成；模型包交付与 Nano 验收另行记录'
                 if set(args.experiments) == {'T', 'S0', 'S1', 'S2', 'S3'} else
                 '所选实验 ' + ', '.join(args.experiments) + ' 已完成；其余实验状态见列表')
        record_progress(args.out, stage)
        return result
    except Exception as error:
        write_json(Path(args.out) / 'pipeline.json', {'status': 'failed', 'error': repr(error)})
        record_progress(args.out, '流水线失败：' + str(error))
        raise


def run_pipeline(args):
    from learning import train, restore_model, evaluate, export_model, write_json, comparison, record_progress
    from data import sha256
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for name in args.experiments:
        if name == 'S1' and args.calibrate:
            record_progress(out, '验证集参数校准')
            calibrate(args, out, device)
        config = config_file(Path(args.config_dir) / (name + '.json'))
        if name == 'T' and args.teacher_run:
            record_progress(out, '核对并复用已完成教师，保留原训练身份')
            reuse_teacher(args.teacher_run, out / 'T', args.manifests, config)
            write_json(out / 'effective_configs/T.json', config)
            continue
        if name in ('S1', 'S2', 'S3') and (out / 'calibration/selection.json').exists():
            selected = config_file(out / 'calibration/selection.json')
            config.update(selected['parameters'])
        write_json(out / 'effective_configs' / (name + '.json'), config)
        run_out = out / name
        run_path = run_out / 'run.json'
        resume = run_path.exists() and config_file(run_path)['status'] != 'completed'
        record = train(config, args.root, args.manifests, run_out, device, resume,
                       out / 'T/best.pt', out / 'S0/best_validation.json', cache=args.cache, validation_cache=args.validation_cache)
        if record['status'] != 'completed':
            raise RuntimeError('Pipeline stopped at incomplete experiment ' + name)
    students = [(config_file(out / name / 'best_validation.json')['miou'], name)
                for name in ('S0', 'S1', 'S2', 'S3') if (out / name / 'best_validation.json').exists()]
    if len(students) == 4:
        selected = max(students)[1]
        write_json(out / 'selection.json', {'selected': selected, 'rule': 'highest validation mIoU; no test-based selection'})
    test_manifest = Path(args.manifests) / 'test.json'
    test_manifest_sha = sha256(test_manifest)
    test_samples = len(config_file(test_manifest)['records'])
    for name in args.experiments:
        run_out = out / name
        record = config_file(run_out / 'run.json')
        config = record['config']
        checkpoint_sha = sha256(run_out / 'best.pt')
        model = None
        for condition in ('clean', 'brightness', 'blur', 'jpeg'):
            evaluation_path = run_out / ('test_' + condition + '.json')
            if evaluation_is_current(evaluation_path, record['identity'], checkpoint_sha, test_manifest_sha, condition, test_samples):
                record_progress(out, '{}：复用已验证测试 {}'.format(name, condition))
                continue
            if model is None:
                model, checkpoint = restore_model(run_out / 'best.pt', device)
                if checkpoint['identity'] != record['identity'] or checkpoint['config'] != config:
                    raise ValueError('Checkpoint/run provenance mismatch: ' + name)
            record_progress(out, '{}：完整测试集 {}'.format(name, condition))
            evaluated = evaluate(model, args.root, test_manifest, config, device, condition,
                                 run_out / 'examples' if condition == 'clean' else None, cache=args.test_cache)
            evaluated.update(checkpoint_sha256=checkpoint_sha,
                             run_identity=record['identity'], smoke=record['smoke'])
            write_json(evaluation_path, evaluated)
        results[name] = {'validation': record['best_miou'],
                         'test': config_file(run_out / 'test_clean.json'), 'identity': record['identity']}
        del model
    if len(students) == 4:
        export_pairs = {(selected, size) for size in (256, 384, 512)} | {('S0', 384)}
        checkpoint_hashes = {name: sha256(out / name / 'best.pt') for name, _ in export_pairs}
        for name, size in sorted(export_pairs):
            model_path = out / 'exports' / ('{}_{}.onnx'.format(name, size))
            if export_is_current(model_path, checkpoint_hashes[name], size):
                record_progress(out, '{}：复用已验证导出 {}'.format(name, size))
                continue
            record_progress(out, '{}：导出与对齐 {}'.format(name, size))
            export_model(out / name / 'best.pt', model_path, size=size)
        record_progress(out, '生成固定案例和失败案例对照图')
        render_comparison(out, selected, args.root, test_manifest, device)
    write_json(out / 'summary.json', results)
    from report import summarize
    summarize(out, out / 'report')
    return results


def calibrate(args, out, device):
    from learning import train, write_json
    from data import sha256
    source = {'teacher': sha256(out / 'T/best.pt'), 'baseline_validation': sha256(out / 'S0/best_validation.json')}
    selection = out / 'calibration/selection.json'
    if selection.exists():
        previous = config_file(selection)
        if previous['source'] != source:
            raise ValueError('Calibration inputs changed; select a new run directory')
        return previous
    trials = []
    candidates = [('boundary_01', 'S1', {'lambda_boundary': 0.1}),
                  ('boundary_02', 'S1', {'lambda_boundary': 0.2}),
                  ('kd_T2_W025', 'S2', {'temperature': 2.0, 'lambda_kd': 0.25}),
                  ('kd_T4_W05', 'S2', {'temperature': 4.0, 'lambda_kd': 0.5})]
    for label, experiment, changes in candidates:
        config = config_file(Path(args.config_dir) / (experiment + '.json'))
        config.update(changes)
        config.update(epochs=3, experiment='calibration_' + label)
        trial_dir = out / 'calibration' / label
        resume = (trial_dir / 'run.json').exists() and config_file(trial_dir / 'run.json')['status'] != 'completed'
        run = train(config, args.root, args.manifests, trial_dir, device, resume,
                    out / 'T/best.pt', out / 'S0/best_validation.json', cache=args.cache, validation_cache=args.validation_cache)
        trials.append({'name': label, 'family': experiment, 'parameters': changes,
                       'validation_miou': run['best_miou'], 'identity': run['identity']})
    parameters = {}
    for family in ('S1', 'S2'):
        winner = max((trial for trial in trials if trial['family'] == family), key=lambda x: x['validation_miou'])
        parameters.update(winner['parameters'])
    record = {'source': source, 'trials': trials, 'parameters': parameters,
              'rule': 'three full epochs per candidate; validation mIoU only; no test evaluation'}
    write_json(selection, record)
    return record


def main():
    parser = argparse.ArgumentParser(description='Transparent semantic segmentation coursework')
    sub = parser.add_subparsers(dest='command', required=True)
    env = sub.add_parser('env')
    env.add_argument('--out', default='artifacts/environment.json')
    synth = sub.add_parser('synthetic')
    synth.add_argument('--root', required=True)
    synth.add_argument('--count', type=int, default=12)
    audit = sub.add_parser('audit')
    audit.add_argument('--root', required=True)
    audit.add_argument('--out', required=True)
    audit.add_argument('--encoding', choices=['trans10k_v1', 'indexed'], required=True)
    audit.add_argument('--synthetic', action='store_true')
    audit.add_argument('--geometry-overrides')
    audit.add_argument('--workers', type=int, default=4)
    splits = sub.add_parser('prepare-splits')
    splits.add_argument('--audit', required=True)
    splits.add_argument('--out', required=True)
    train = sub.add_parser('train')
    train.add_argument('--config', required=True)
    train.add_argument('--root', required=True)
    train.add_argument('--manifests', required=True)
    train.add_argument('--out', required=True)
    train.add_argument('--device', default='auto')
    train.add_argument('--resume', action='store_true')
    train.add_argument('--teacher')
    train.add_argument('--baseline-validation')
    train.add_argument('--stop-after-epoch', type=int, default=0)
    train.add_argument('--max-train-steps', type=int, default=0)
    train.add_argument('--cache')
    train.add_argument('--validation-cache')
    cache = sub.add_parser('cache')
    cache.add_argument('--root', required=True)
    cache.add_argument('--manifest', required=True)
    cache.add_argument('--out', required=True)
    cache.add_argument('--size', type=int, default=384)
    cache.add_argument('--conditions', nargs='+', choices=['clean', 'brightness', 'blur', 'jpeg'], default=['clean'])
    cache.add_argument('--seed', type=int, default=42)
    cache.add_argument('--workers', type=int, default=4)
    targets = sub.add_parser('cache-targets')
    targets.add_argument('--root', required=True)
    targets.add_argument('--manifest', required=True)
    targets.add_argument('--cache', required=True)
    targets.add_argument('--workers', type=int, default=4)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--root', required=True)
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--corruption', choices=['clean', 'brightness', 'blur', 'jpeg'], default='clean')
    evaluate.add_argument('--cache')
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', required=True)
    export.add_argument('--out', required=True)
    export.add_argument('--opset', type=int, choices=[11, 12], default=11)
    export.add_argument('--size', type=int, choices=[64, 96, 128, 256, 384, 512])
    export.add_argument('--smoke', action='store_true')
    demo = sub.add_parser('demo')
    demo.add_argument('--model', required=True)
    demo.add_argument('--source', required=True)
    demo.add_argument('--out', required=True)
    demo.add_argument('--backend', choices=['torch', 'onnx', 'tensorrt'], default='onnx')
    demo.add_argument('--device', default='cpu')
    demo.add_argument('--show', action='store_true')
    demo.add_argument('--async-capture', action='store_true')
    demo.add_argument('--smoke', action='store_true')
    demo.add_argument('--duration', type=float, default=0)
    demo.add_argument('--max-frames', type=int, default=0)
    demo.add_argument('--loop', action='store_true')
    compare = sub.add_parser('compare')
    compare.add_argument('--baseline', required=True)
    compare.add_argument('--final', required=True)
    compare.add_argument('--root', required=True)
    compare.add_argument('--manifest', required=True)
    compare.add_argument('--out', required=True)
    compare.add_argument('--device', default='cpu')
    suite = sub.add_parser('pipeline')
    suite.add_argument('--root', required=True)
    suite.add_argument('--manifests', required=True)
    suite.add_argument('--out', required=True)
    suite.add_argument('--config-dir', default='configs')
    suite.add_argument('--experiments', nargs='+', choices=['T', 'S0', 'S1', 'S2', 'S3'], default=['T', 'S0', 'S1', 'S2', 'S3'])
    suite.add_argument('--device', default='auto')
    suite.add_argument('--cache')
    suite.add_argument('--validation-cache')
    suite.add_argument('--test-cache')
    suite.add_argument('--calibrate', action='store_true')
    suite.add_argument('--teacher-run', help='Reuse a complete teacher with matching config/data and preserved provenance')
    args = parser.parse_args()
    if args.command == 'env':
        result = environment(args.out)
    elif args.command == 'synthetic':
        from data import synthetic
        synthetic(args.root, args.count)
        result = {'source': 'synthetic-smoke', 'root': args.root}
    elif args.command == 'audit':
        from data import audit
        result = audit(args.root, args.out, args.encoding, not args.synthetic, args.geometry_overrides, args.workers)
    elif args.command == 'prepare-splits':
        from data import prepare_splits
        result = prepare_splits(args.audit, args.out)
    elif args.command == 'train':
        from learning import train
        result = train(config_file(args.config), args.root, args.manifests, args.out, args.device, args.resume,
                       args.teacher, args.baseline_validation, args.stop_after_epoch, args.max_train_steps, args.cache, args.validation_cache)
    elif args.command == 'cache':
        from data import prepare_cache
        result = prepare_cache(args.root, args.manifest, args.out, args.size, args.conditions, args.seed, args.workers)
    elif args.command == 'cache-targets':
        from data import cache_original_targets
        result = cache_original_targets(args.root, args.manifest, args.cache, args.workers)
    elif args.command == 'evaluate':
        from learning import restore_model, evaluate, write_json
        model, cp = restore_model(args.checkpoint, args.device)
        result = evaluate(model, args.root, args.manifest, cp['config'], args.device, args.corruption, cache=args.cache)
        result['smoke'] = cp['smoke']
        write_json(args.out, result)
    elif args.command == 'export':
        from learning import export_model
        result = export_model(args.checkpoint, args.out, args.opset, args.size, args.smoke)
    elif args.command == 'demo':
        from runtime import demo, ONNXRunner
        from nano import source_label
        if args.backend == 'torch':
            from learning import TorchRunner
            runner = TorchRunner(args.model, args.device)
            if runner.checkpoint['smoke'] and not args.smoke:
                raise ValueError('Smoke checkpoint requires --smoke')
            if runner.checkpoint['config']['kind'] == 'student':
                runner.model.boundary_head = None
        elif args.backend == 'onnx':
            metadata = config_file(Path(args.model).with_suffix('.json'))
            if metadata['smoke'] and not args.smoke:
                raise ValueError('Smoke ONNX requires --smoke')
            runner = ONNXRunner(args.model)
        else:
            from nano import TensorRTRunner
            runner = TensorRTRunner(args.model)
            if runner.metadata['smoke'] and not args.smoke:
                runner.close()
                raise ValueError('Smoke engine requires --smoke')
        try:
            result = demo(runner, args.source, args.out, not args.show, args.async_capture, duration=args.duration,
                          max_frames=args.max_frames, label=source_label(args.source) + ' ' + Path(args.model).stem,
                          smoke=args.smoke, loop=args.loop)
        finally:
            runner.close()
    elif args.command == 'compare':
        from learning import comparison
        result = comparison(args.baseline, args.final, args.root, args.manifest, args.out, args.device)
    else:
        result = pipeline(args)
    if args.command == 'pipeline':
        result = {'status': 'completed', 'out': args.out, 'experiments': list(result)}
    elif args.command == 'train':
        result = {key: result.get(key) for key in ('status', 'global_step', 'best_miou', 'elapsed_seconds', 'identity')}
    elif args.command == 'evaluate':
        result = {key: value for key, value in result.items() if key != 'per_image'}
    elif args.command == 'audit':
        result = {key: result[key] for key in ('passed', 'counts', 'errors', 'journal_reused')}
    elif args.command == 'prepare-splits':
        result = {key: result[key] for key in ('passed', 'protocol', 'original_counts', 'counts', 'excluded')}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
