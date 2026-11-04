from contextlib import nullcontext
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

from data import SegmentationData, sha256
from metrics import Metrics, corrupt
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


def identity(config, manifests):
    sources = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('local.')}
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


def evaluate(model, root, manifest, config, device, corruption='clean', preview=None, max_samples=0):
    data = SegmentationData(root, manifest, config['size'])
    metrics = Metrics(config['boundary_tolerance'])
    per_image = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for i in range(min(len(data), max_samples) if max_samples else len(data)):
            rgb, target = data.original(i)
            image = corrupt(rgb, corruption, config['seed'] + i)
            tensor, meta = preprocess(image, config['size'], config.get('mean', MEAN), config.get('std', STD))
            logits = model(torch.from_numpy(tensor).to(device))['logits'].float().cpu().numpy()
            prediction = restore_logits(logits, meta).argmax(-1).astype(np.uint8)
            metrics.update(prediction, target)
            single = Metrics(config['boundary_tolerance'])
            single.update(prediction, target)
            per_image.append({'image': data.records[i]['image'], 'miou': single.result()['miou']})
            if preview and i < 4:
                preview = Path(preview)
                preview.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.concatenate([rgb, overlay(rgb, target), overlay(rgb, prediction)], axis=1)).save(preview / ('{:03}.png'.format(i)))
    model.train(was_training)
    result = metrics.result()
    result.update({'samples': len(per_image), 'corruption': corruption, 'seed': config['seed'],
                   'dataset_source': data.manifest['source'], 'split': data.manifest['split'], 'per_image': per_image})
    return result


def renormalize(image, source, target):
    source_mean = torch.tensor(source.get('mean', MEAN), device=image.device).view(1, 3, 1, 1)
    source_std = torch.tensor(source.get('std', STD), device=image.device).view(1, 3, 1, 1)
    target_mean = torch.tensor(target.get('mean', MEAN), device=image.device).view(1, 3, 1, 1)
    target_std = torch.tensor(target.get('std', STD), device=image.device).view(1, 3, 1, 1)
    return (image * source_std + source_mean - target_mean) / target_std


def train(config, root, manifests, out, device='auto', resume=False, teacher_path=None, baseline_path=None,
          stop_after_epoch=0, max_train_steps=0):
    out, manifests = Path(out), Path(manifests)
    out.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = dict(config)
    smoke = config.get('smoke', False)
    train_manifest, val_manifest = manifests / 'train.json', manifests / 'validation.json'
    train_data = SegmentationData(root, train_manifest, config['size'], config['augment'], config['seed'])
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
    use_amp = config['amp'] and device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
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
           'device': device, 'provenance': provenance}
    write_json(out / 'run.json', run)
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for epoch in range(epoch_start, config['epochs']):
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
                with torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext():
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                losses.append(loss.item())
                step += 1
                if max_train_steps and index + 1 >= max_train_steps:
                    break
            scheduler.step()
            validation = evaluate(model, root, val_manifest, config, device)
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
                     'epoch_seconds': time.perf_counter() - epoch_start_time, 'lr_next': scheduler.get_last_lr()[0]}
            with (out / 'epochs.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(entry) + '\n')
            print(json.dumps(entry), flush=True)
            if stop_after_epoch and epoch + 1 >= stop_after_epoch and epoch + 1 < config['epochs']:
                run['status'] = 'interrupted_at_epoch_boundary'
                break
        else:
            run['status'] = 'completed'
    except Exception as error:
        run.update(status='failed', error=repr(error))
        write_json(out / 'run.json', run)
        raise
    run.update(global_step=step, best_miou=best, elapsed_seconds=time.perf_counter() - started,
               peak_cuda_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None)
    write_json(out / 'run.json', run)
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
    record = {'checkpoint_sha256': sha256(checkpoint_path), 'onnx_sha256': sha256(out), 'bytes': out.stat().st_size,
              'opset': opset, 'shape': [1, 3, size, size], 'output': 'three-class float logits',
              'smoke': checkpoint['smoke'], 'auxiliary_removed_exact': True, 'comparisons': comparisons,
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
