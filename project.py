import argparse
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


def pipeline(args):
    from learning import train, restore_model, evaluate, export_model, write_json, comparison
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {}
    for name in args.experiments:
        config = config_file(Path(args.config_dir) / (name + '.json'))
        run_out = out / name
        run_path = run_out / 'run.json'
        resume = run_path.exists() and config_file(run_path)['status'] != 'completed'
        record = train(config, args.root, args.manifests, run_out, device, resume,
                       out / 'T/best.pt', out / 'S0/best_validation.json')
        if record['status'] != 'completed':
            raise RuntimeError('Pipeline stopped at incomplete experiment ' + name)
        model, checkpoint = restore_model(run_out / 'best.pt', device)
        for condition in ('clean', 'brightness', 'blur', 'jpeg'):
            evaluated = evaluate(model, args.root, Path(args.manifests) / 'test.json', config, device, condition,
                                 run_out / 'examples' if condition == 'clean' else None)
            evaluated.update(checkpoint_sha256=__import__('data').sha256(run_out / 'best.pt'),
                             run_identity=record['identity'], smoke=record['smoke'])
            write_json(run_out / ('test_' + condition + '.json'), evaluated)
        results[name] = {'validation': checkpoint['validation']['miou'],
                         'test': config_file(run_out / 'test_clean.json'), 'identity': record['identity']}
        del model
    students = [(config_file(out / name / 'best_validation.json')['miou'], name)
                for name in ('S0', 'S1', 'S2', 'S3') if (out / name / 'best_validation.json').exists()]
    if len(students) == 4:
        selected = max(students)[1]
        write_json(out / 'selection.json', {'selected': selected, 'rule': 'highest validation mIoU; no test-based selection'})
        for size in (256, 384, 512):
            export_model(out / selected / 'best.pt', out / 'exports' / ('{}_{}.onnx'.format(selected, size)), size=size)
        comparison(out / 'S0/best.pt', out / selected / 'best.pt', args.root,
                   Path(args.manifests) / 'test.json', out / 'comparison', device)
    write_json(out / 'summary.json', results)
    return results


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
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', required=True)
    evaluate.add_argument('--root', required=True)
    evaluate.add_argument('--manifest', required=True)
    evaluate.add_argument('--out', required=True)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--corruption', choices=['clean', 'brightness', 'blur', 'jpeg'], default='clean')
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
    args = parser.parse_args()
    if args.command == 'env':
        result = environment(args.out)
    elif args.command == 'synthetic':
        from data import synthetic
        synthetic(args.root, args.count)
        result = {'source': 'synthetic-smoke', 'root': args.root}
    elif args.command == 'audit':
        from data import audit
        result = audit(args.root, args.out, args.encoding, not args.synthetic)
    elif args.command == 'train':
        from learning import train
        result = train(config_file(args.config), args.root, args.manifests, args.out, args.device, args.resume,
                       args.teacher, args.baseline_validation, args.stop_after_epoch, args.max_train_steps)
    elif args.command == 'evaluate':
        from learning import restore_model, evaluate, write_json
        model, cp = restore_model(args.checkpoint, args.device)
        result = evaluate(model, args.root, args.manifest, cp['config'], args.device, args.corruption)
        result['smoke'] = cp['smoke']
        write_json(args.out, result)
    elif args.command == 'export':
        from learning import export_model
        result = export_model(args.checkpoint, args.out, args.opset, args.size, args.smoke)
    elif args.command == 'demo':
        from runtime import demo, ONNXRunner
        if args.backend == 'torch':
            from learning import TorchRunner
            runner = TorchRunner(args.model, args.device)
            if runner.checkpoint['smoke'] and not args.smoke:
                raise ValueError('Smoke checkpoint requires --smoke')
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
                          max_frames=args.max_frames, label=Path(args.model).stem, smoke=args.smoke, loop=args.loop)
        finally:
            runner.close()
    elif args.command == 'compare':
        from learning import comparison
        result = comparison(args.baseline, args.final, args.root, args.manifest, args.out, args.device)
    else:
        result = pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
