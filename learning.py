from contextlib import nullcontext
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader
from filelock import FileLock

from data import SegmentationData, sha256
from metrics import Metrics, corrupt, tensor_statistics, miou_from_confusion
from models import build_model, segmentation_loss, boundary_loss, distillation_loss
from runtime import preprocess, restore_logits, overlay, MEAN, STD


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def record_progress(run_root, stage, document=None):
    run_root = Path(run_root)
    document = Path(document) if document else Path(__file__).parent / 'docs/PROGRESS.md'
    stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')
    rows = []
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        path = run_root / name / 'run.json'
        value = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
        rows.append({'name': name, 'status': value.get('status', 'pending'),
                     'epoch': value.get('last_completed_epoch', 0), 'budget': value.get('config', {}).get('epochs'),
                     'best_validation_miou': value.get('best_miou'), 'pid': value.get('pid')})
    snapshot = {'updated_at': stamp, 'run_root': str(run_root), 'stage': stage, 'experiments': rows}
    write_json(run_root / 'CURRENT_STAGE.json', snapshot)
    begin, end = '<!-- CV_CW2_AUTO_PROGRESS_BEGIN -->', '<!-- CV_CW2_AUTO_PROGRESS_END -->'
    lines = [begin, '## 自动运行进度', '', '更新时间：' + stamp, '当前阶段：' + stage,
             '运行目录：' + str(run_root), '以下为最近检查点；恢复前仍需核对进程和日志。', '']
    for row in rows:
        score = row['best_validation_miou']
        text = '{}：{}，完整轮次 {}/{}'.format(row['name'], row['status'], row['epoch'], row['budget'] or 'pending')
        if score is not None:
            text += '，最佳验证 mIoU {:.6f}'.format(score)
        lines.append(text + '。')
    lines.extend(['', '模型训练、测试、导出与真实 Nano 验收是不同状态。', end, ''])
    generated = '\n'.join(lines)
    document.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(document) + '.lock', timeout=10):
        original = document.read_text(encoding='utf-8') if document.exists() else ''
        if begin in original and end in original:
            before, _, rest = original.partition(begin)
            _, _, after = rest.partition(end)
            updated = before + generated + after.lstrip('\r\n')
        else:
            updated = generated + '\n' + original
        temporary = document.with_suffix('.md.tmp')
        temporary.write_text(updated, encoding='utf-8')
        temporary.replace(document)


def identity(config, manifests):
    sources = {name: sha256(Path(__file__).parent / name) for name in ('data.py', 'models.py', 'metrics.py', 'learning.py', 'runtime.py')}
    records = {Path(p).name: sha256(p) for p in manifests}
    combined = {'config': config, 'sources': sources, 'manifests': records}
    digest = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()
    return digest, combined


def load_checkpoint(path, device='cpu'):
    return torch.load(path, map_location=device, weights_only=False)


def restore_model(path, device='cpu'):
    checkpoint = load_checkpoint(path)
    model = build_model(checkpoint['config'], pretrained=False)
    model.load_state_dict(checkpoint['model'])
    return model.to(device).eval(), checkpoint


def evaluate(model, root, manifest, config, device, corruption='clean', preview=None, max_samples=0, cache=None):
    data = SegmentationData(root, manifest, config['size'], seed=config['seed'], mean=config.get('mean'), std=config.get('std'), cache=cache)
    metrics = Metrics(config['boundary_tolerance'])
    per_image = []
    gpu_metric_samples = 0
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(min(len(data), max_samples) if max_samples else len(data)):
            tensor, meta, target = data.evaluation_item(i, corruption)
            logits = model(torch.from_numpy(tensor).to(device))['logits'].float().cpu().numpy()
            restored = restore_logits(logits, meta)
            if device.startswith('cuda') and target.size <= 17000000:
                prediction_gpu = torch.from_numpy(restored).to(device).argmax(-1).to(torch.uint8)
                target_array = np.asarray(target)
                if not target_array.flags.writeable:
                    target_array = target_array.copy()
                target_gpu = torch.from_numpy(target_array).to(device)
                counts, boundary_counts = tensor_statistics(prediction_gpu, target_gpu, config['boundary_tolerance'])
                metrics.confusion += counts
                metrics.boundary += boundary_counts
                gpu_metric_samples += 1
                prediction = prediction_gpu.cpu().numpy() if preview and i < 4 else None
                del prediction_gpu, target_gpu
            else:
                prediction = restored.argmax(-1).astype(np.uint8)
                counts = metrics.update(prediction, target)
            per_image.append({'image': data.records[i]['image'], 'miou': miou_from_confusion(counts)})
            if preview and i < 4:
                rgb, _ = data.original(i)
                preview = Path(preview)
                preview.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.concatenate([rgb, overlay(rgb, target), overlay(rgb, prediction)], axis=1)).save(preview / ('{:03}.png'.format(i)))
    model.train(was_training)
    result = metrics.result()
    result.update({'samples': len(per_image), 'corruption': corruption, 'seed': config['seed'],
                   'dataset_source': data.manifest['source'], 'split': data.manifest['split'], 'per_image': per_image,
                   'manifest_sha256': sha256(manifest), 'data_protocol': data.manifest.get('protocol', 'official'),
                   'effective_split_counts': data.manifest.get('effective_counts'), 'gpu_metric_samples': gpu_metric_samples})
    return result


def renormalize(image, source, target):
    source_mean = torch.tensor(source.get('mean', MEAN), device=image.device).view(1, 3, 1, 1)
    source_std = torch.tensor(source.get('std', STD), device=image.device).view(1, 3, 1, 1)
    target_mean = torch.tensor(target.get('mean', MEAN), device=image.device).view(1, 3, 1, 1)
    target_std = torch.tensor(target.get('std', STD), device=image.device).view(1, 3, 1, 1)
    return (image * source_std + source_mean - target_mean) / target_std


def amp_settings(config, device):
    name = config.get('amp_dtype', 'float16')
    if name not in ('float16', 'bfloat16'):
        raise ValueError('amp_dtype must be float16 or bfloat16')
    enabled = config['amp'] and device.startswith('cuda')
    if enabled and name == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This CUDA device does not support the configured bfloat16 training')
    return enabled, getattr(torch, name), enabled and name == 'float16'


def train(config, root, manifests, out, device='auto', resume=False, teacher_path=None, baseline_path=None,
          stop_after_epoch=0, max_train_steps=0, cache=None, validation_cache=None):
    Path(out).mkdir(parents=True, exist_ok=True)
    with FileLock(str(Path(out) / '.train.lock'), timeout=0):
        return _train(config, root, manifests, out, device, resume, teacher_path, baseline_path,
                      stop_after_epoch, max_train_steps, cache, validation_cache)


def _train(config, root, manifests, out, device='auto', resume=False, teacher_path=None, baseline_path=None,
           stop_after_epoch=0, max_train_steps=0, cache=None, validation_cache=None):
    out, manifests = Path(out), Path(manifests)
    out.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = dict(config)
    smoke = config.get('smoke', False)
    train_manifest, val_manifest = manifests / 'train.json', manifests / 'validation.json'
    train_data = SegmentationData(root, train_manifest, config['size'], config['augment'], config['seed'],
                                  config.get('mean'), config.get('std'), cache)
    if not smoke and train_data.manifest['source'] != 'Trans10K v1':
        raise ValueError('Formal training requires audited real Trans10K v1')
    if not smoke and device == 'cpu':
        raise ValueError('Formal training on CPU is disabled; use explicit smoke or a CUDA device')
    if max_train_steps and not smoke:
        raise ValueError('Step truncation is only allowed for smoke runs')
    run_id, provenance = identity(config, [train_manifest, val_manifest])
    if (out / 'run.json').exists():
        previous = json.loads((out / 'run.json').read_text())
        if previous['identity'] != run_id:
            raise ValueError('Run configuration/code/data changed; choose a new output directory')
        if previous['status'] == 'completed' and not resume:
            return previous
        if not resume:
            raise ValueError('Incomplete run exists; use --resume or a new directory')
    seed_everything(config['seed'])
    model = build_model(config, pretrained=False if resume else None).to(device)
    teacher = teacher_config = None
    if config['distill']:
        if not teacher_path or not Path(teacher_path).is_file():
            raise ValueError('Distillation requires a trained teacher checkpoint')
        teacher, checkpoint = restore_model(teacher_path, device)
        teacher_config = checkpoint['config']
        if not smoke:
            if checkpoint['smoke'] or checkpoint['epoch'] < 1 or not baseline_path:
                raise ValueError('Formal KD requires trained teacher and S0 validation evidence')
            baseline = json.loads(Path(baseline_path).read_text())
            if checkpoint['validation']['miou'] <= baseline['miou']:
                raise ValueError('Teacher is not better than S0 on validation; inspect training before KD')
            if checkpoint['validation_manifest_sha256'] != sha256(val_manifest):
                raise ValueError('Teacher validation split differs')
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    batch = config['batch_size']
    if batch == 'auto':
        batch = 4 if device.startswith('cuda') and torch.cuda.get_device_properties(device).total_memory >= 7 * 1024 ** 3 else 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    use_amp, amp_dtype, use_scaler = amp_settings(config, device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)
    epoch_start, step, best = 0, 0, -1.0
    if resume:
        checkpoint = load_checkpoint(out / 'last.pt')
        if checkpoint['identity'] != run_id:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        epoch_start, step, best = checkpoint['epoch'], checkpoint['global_step'], checkpoint['best_miou']
        random.setstate(checkpoint['random_state'])
        np.random.set_state(checkpoint['numpy_state'])
        torch.set_rng_state(checkpoint['torch_state'])
        if device.startswith('cuda') and checkpoint['cuda_state']:
            torch.cuda.set_rng_state_all(checkpoint['cuda_state'])
    run = {'identity': run_id, 'status': 'running', 'smoke': smoke, 'config': config, 'batch_size': batch,
           'device': device, 'provenance': provenance, 'pid': os.getpid(),
           'last_completed_epoch': epoch_start, 'global_step': step, 'best_miou': best if best >= 0 else None}
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    run['git_commit_at_start'] = git.stdout.strip() if git.returncode == 0 else 'unavailable'
    write_json(out / 'run.json', run)
    progress_root = out.parent.parent if config.get('experiment', '').startswith('calibration_') else out.parent
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for epoch in range(epoch_start, config['epochs']):
            if not smoke:
                record_progress(progress_root, '{}：训练第 {} 轮'.format(config['experiment'], epoch + 1))
            train_data.epoch = epoch
            generator = torch.Generator().manual_seed(config['seed'] + epoch)
            loader = DataLoader(train_data, batch_size=batch, shuffle=True, num_workers=0,
                                generator=generator, drop_last=False)
            model.train()
            if config.get('freeze_bn', True):
                for module in model.modules():
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
            losses, epoch_start_time = [], time.perf_counter()
            for index, (image, mask) in enumerate(loader):
                image, mask = image.to(device), mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else nullcontext():
                    output = model(image)
                    loss = segmentation_loss(output['logits'], mask)
                    if config['boundary']:
                        loss = loss + config['lambda_boundary'] * boundary_loss(output['boundary'], mask, config['boundary_radius'])
                    if config['distill']:
                        with torch.no_grad():
                            teacher_output = teacher(renormalize(image, config, teacher_config))['logits']
                        loss = loss + config['lambda_kd'] * distillation_loss(output['logits'], teacher_output, mask, config['temperature'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=not use_scaler)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                step += 1
                if max_train_steps and index + 1 >= max_train_steps:
                    break
            scheduler.step()
            training_seconds = time.perf_counter() - epoch_start_time
            if not smoke:
                record_progress(progress_root, '{}：验证第 {} 轮'.format(config['experiment'], epoch + 1))
            validation_started = time.perf_counter()
            validation = evaluate(model, root, val_manifest, config, device, cache=validation_cache)
            validation_seconds = time.perf_counter() - validation_started
            score = validation['miou']
            if score is None:
                raise ValueError('Validation contains no valid pixels')
            improved = score > best
            best = max(best, score)
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'global_step': step, 'best_miou': best,
                          'config': config, 'identity': run_id, 'smoke': smoke, 'validation': validation,
                          'validation_manifest_sha256': sha256(val_manifest), 'random_state': random.getstate(),
                          'numpy_state': np.random.get_state(), 'torch_state': torch.get_rng_state(),
                          'cuda_state': torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
            torch.save(checkpoint, out / 'last.pt.tmp')
            (out / 'last.pt.tmp').replace(out / 'last.pt')
            if improved:
                torch.save(checkpoint, out / 'best.pt.tmp')
                (out / 'best.pt.tmp').replace(out / 'best.pt')
                write_json(out / 'best_validation.json', validation)
            entry = {'epoch': epoch + 1, 'step': step, 'loss': float(np.mean(losses)), 'validation_miou': score,
                     'epoch_seconds': time.perf_counter() - epoch_start_time, 'train_seconds': training_seconds,
                     'validation_seconds': validation_seconds, 'lr_next': scheduler.get_last_lr()[0]}
            with (out / 'epochs.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(entry) + '\n')
            run.update(last_completed_epoch=epoch + 1, global_step=step, best_miou=best)
            write_json(out / 'run.json', run)
            print(json.dumps(entry), flush=True)
            if stop_after_epoch and epoch + 1 >= stop_after_epoch and epoch + 1 < config['epochs']:
                run['status'] = 'interrupted_at_epoch_boundary'
                break
        else:
            run['status'] = 'completed'
    except Exception as error:
        run.update(status='failed', error=repr(error))
        write_json(out / 'run.json', run)
        if not smoke:
            record_progress(progress_root, '{}：失败，查看 run.json 与控制台日志'.format(config['experiment']))
        raise
    run.update(global_step=step, best_miou=best, elapsed_seconds=time.perf_counter() - started,
               peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    write_json(out / 'run.json', run)
    if not smoke:
        record_progress(progress_root, '{}：{}'.format(config['experiment'], run['status']))
    return run


class TorchRunner:
    def __init__(self, checkpoint, device='cpu'):
        self.model, self.checkpoint = restore_model(checkpoint, device)
        self.device, self.size = device, self.checkpoint['config']['size']
        self.mean = self.checkpoint['config'].get('mean', MEAN)
        self.std = self.checkpoint['config'].get('std', STD)

    def __call__(self, tensor):
        image = torch.from_numpy(tensor).to(self.device)
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(image)['logits']
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000
        return logits.float().cpu().numpy(), elapsed

    def close(self):
        self.model = None


def export_model(checkpoint_path, out, opset=11, size=None, smoke=False):
    import onnx
    import onnxruntime as ort
    model, checkpoint = restore_model(checkpoint_path)
    if checkpoint['config']['kind'] != 'student':
        raise ValueError('Only students are deployment targets')
    if checkpoint['smoke'] and not smoke:
        raise ValueError('Smoke checkpoint requires explicit --smoke')
    size = size or checkpoint['config']['size']
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint['smoke'] and 'smoke' not in out.stem.lower():
        raise ValueError('Smoke ONNX filename must include smoke')
    inference = model.inference_model()
    seed_everything(123)
    example = torch.randn(1, 3, size, size)
    with torch.no_grad():
        main = model(example)['logits']
        stripped = inference(example)
    torch.testing.assert_close(main, stripped, rtol=0, atol=0)
    torch.onnx.export(inference, example, str(out), opset_version=opset, input_names=['image'],
                      output_names=['logits'], dynamic_axes=None, do_constant_folding=True, dynamo=False)
    graph = onnx.load(str(out))
    onnx.checker.check_model(graph)
    session = ort.InferenceSession(str(out), providers=['CPUExecutionProvider'])
    comparisons = []
    reference_inputs, reference_logits = [], []
    for tensor in (example, torch.zeros_like(example), torch.rand_like(example) * 4 - 2):
        with torch.no_grad():
            reference = inference(tensor).numpy()
        actual = session.run(None, {'image': tensor.numpy()})[0]
        difference = np.abs(actual - reference)
        np.testing.assert_allclose(actual, reference, rtol=2e-3, atol=2e-4)
        agreement = float((actual.argmax(1) == reference.argmax(1)).mean())
        if agreement < 0.999:
            raise AssertionError('Mask agreement below 99.9%')
        comparisons.append({'max_abs': float(difference.max()), 'mean_abs': float(difference.mean()), 'mask_agreement': agreement})
        reference_inputs.append(tensor.numpy())
        reference_logits.append(reference)
    reference_path = out.with_suffix('.reference.npz')
    np.savez_compressed(reference_path, inputs=np.stack(reference_inputs), logits=np.stack(reference_logits))
    record = {'checkpoint_sha256': sha256(checkpoint_path), 'onnx_sha256': sha256(out), 'bytes': out.stat().st_size,
              'opset': opset, 'shape': [1, 3, size, size], 'output': 'three-class float logits',
              'smoke': checkpoint['smoke'], 'auxiliary_removed_exact': True, 'comparisons': comparisons,
              'reference_sha256': sha256(reference_path), 'reference_file': reference_path.name,
              'normalization': {'mean': checkpoint['config'].get('mean', MEAN),
                                'std': checkpoint['config'].get('std', STD)}, 'nano_verified': False}
    write_json(out.with_suffix('.json'), record)
    return record


def comparison(baseline_path, final_path, root, manifest, out, device='cpu', limit=6):
    first, first_cp = restore_model(baseline_path, device)
    last, last_cp = restore_model(final_path, device)
    dataset = SegmentationData(root, manifest, first_cp['config']['size'])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    with torch.no_grad():
        for i in range(len(dataset)):
            rgb, target = dataset.original(i)
            preds = []
            for model, cp in [(first, first_cp), (last, last_cp)]:
                tensor, meta = preprocess(rgb, cp['config']['size'])
                values = model(torch.from_numpy(tensor).to(device))['logits'].cpu().numpy()
                preds.append(restore_logits(values, meta).argmax(-1).astype(np.uint8))
            metric = Metrics()
            metric.update(preds[1], target)
            score = metric.result()['miou']
            scores.append((score if score is not None else 1, i))
            if i < limit:
                panel = np.concatenate([rgb, overlay(rgb, target), overlay(rgb, preds[0]), overlay(rgb, preds[1])], axis=1)
                Image.fromarray(panel).save(out / ('fixed_{:03}.png'.format(i)))
    selected = [i for _, i in sorted(scores)[:limit]]
    for i in selected:
        rgb, target = dataset.original(i)
        panels = [rgb, overlay(rgb, target)]
        for model, cp in [(first, first_cp), (last, last_cp)]:
            tensor, meta = preprocess(rgb, cp['config']['size'])
            with torch.no_grad():
                values = model(torch.from_numpy(tensor).to(device))['logits'].cpu().numpy()
            panels.append(overlay(rgb, restore_logits(values, meta).argmax(-1).astype(np.uint8)))
        Image.fromarray(np.concatenate(panels, axis=1)).save(out / ('failure_{:03}.png'.format(i)))
    write_json(out / 'selection.json', {'fixed_indices': list(range(min(limit, len(dataset)))),
                                      'failure_indices': selected, 'failure_rule': 'lowest final-model per-image mIoU; tie by index',
                                      'columns': ['original', 'ground_truth', 'S0', 'final'],
                                      'smoke': first_cp['smoke'] or last_cp['smoke']})
