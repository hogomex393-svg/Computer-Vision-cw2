import ast
import copy
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from data import load_mask, synthetic, audit, SegmentationData, prepare_cache
from learning import seed_everything, train, load_checkpoint, export_model
from metrics import Metrics, corrupt
from models import Student, Teacher, segmentation_loss, boundary_targets, boundary_loss, distillation_loss
from runtime import letterbox, preprocess, restore_logits, ONNXRunner, demo, video_path, write_image

torch.set_num_threads(2)
Path('artifacts').mkdir(exist_ok=True)


def smoke_config(epochs=2):
    config = json.loads(Path('configs/smoke.json').read_text())
    config.update(epochs=epochs, batch_size=2, decoder_channels=16)
    return config


def test_raw_white_is_stuff_and_indexed_white_is_ignore(tmp_path):
    raw = np.array([[[0, 0, 0], [255, 0, 0], [255, 255, 255]]], np.uint8)
    Image.fromarray(raw).save(tmp_path / 'raw.png')
    np.testing.assert_array_equal(load_mask(tmp_path / 'raw.png', 'trans10k_v1'), [[0, 1, 2]])
    Image.fromarray(np.array([[0, 1, 2, 255]], np.uint8)).save(tmp_path / 'indexed.png')
    np.testing.assert_array_equal(load_mask(tmp_path / 'indexed.png', 'indexed'), [[0, 1, 2, 255]])
    raw[0, 1] = [128, 128, 128]
    Image.fromarray(raw).save(tmp_path / 'bad.png')
    with pytest.raises(ValueError, match='Unknown'):
        load_mask(tmp_path / 'bad.png', 'trans10k_v1')


def test_letterbox_mask_and_restore_coordinates():
    rgb = np.zeros((32, 96, 3), np.uint8)
    mask = np.zeros((32, 96), np.uint8)
    mask[:, 48:] = 2
    rgb[:, 48:] = (255, 0, 0)
    canvas, resized, meta = letterbox(rgb, 96, mask)
    assert np.all(resized[:32] == 255)
    assert canvas[40, 60, 0] == 255 and resized[40, 60] == 2
    logits = np.stack([(resized == c).astype(np.float32) * 10 for c in range(3)])
    prediction = restore_logits(logits, meta).argmax(-1)
    np.testing.assert_array_equal(prediction, mask)
    tensor, _ = preprocess(rgb, 96)
    assert tensor.shape == (1, 3, 96, 96) and tensor.dtype == np.float32


def test_metrics_manual_and_absent_classes():
    metric = Metrics(0)
    metric.update(np.array([[0, 2], [2, 99]]), np.array([[0, 1], [2, 255]]))
    result = metric.result()
    assert result['per_class_iou'] == [1.0, 0.0, 0.5]
    assert result['miou'] == 0.5
    assert result['foreground_iou'] == result['foreground_dice'] == 1
    absent = Metrics()
    absent.update(np.zeros((5, 5), int), np.zeros((5, 5), int))
    assert absent.result()['miou'] == 1 and absent.result()['per_class_iou'][1:] == [None, None]
    empty = Metrics()
    empty.update(np.zeros((2, 2), int), np.full((2, 2), 255))
    assert empty.result()['miou'] is None


def test_boundary_tolerance_and_ignore_neighborhood():
    target = np.zeros((20, 20), np.uint8)
    target[:, 10:] = 1
    prediction = np.zeros_like(target)
    prediction[:, 11:] = 1
    relaxed, exact = Metrics(2), Metrics(0)
    relaxed.update(prediction, target)
    exact.update(prediction, target)
    assert relaxed.result()['boundary_f1'] == 1
    assert exact.result()['boundary_f1'] < 1
    mask = torch.from_numpy(target.astype(np.int64))[None]
    mask[:, :, 9:12] = 255
    edges, valid = boundary_targets(mask, 1)
    assert not edges[valid].any()
    assert not valid[:, :, 7:14].any()


@pytest.mark.parametrize('value', [0, 1, 255])
def test_loss_degenerate_and_ignored_gradients(value):
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    mask = torch.full((2, 8, 8), value, dtype=torch.long)
    boundary = torch.randn(2, 1, 8, 8, requires_grad=True)
    teacher = torch.randn_like(logits, requires_grad=True)
    loss = segmentation_loss(logits, mask) + boundary_loss(boundary, mask, 1) + distillation_loss(logits, teacher, mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert teacher.grad is None
    assert torch.isfinite(logits.grad).all()
    if value == 255:
        assert loss.item() == 0 and logits.grad.abs().sum() == 0


def test_student_scales_initialization_and_auxiliary_removal():
    seed_everything(42)
    plain = Student(16, False, False).eval()
    seed_everything(42)
    boundary = Student(16, True, False).eval()
    for key, parameter in plain.state_dict().items():
        torch.testing.assert_close(parameter, boundary.state_dict()[key], rtol=0, atol=0)
    assert plain.stage_channels == [24, 40, 112, 960]
    assert plain.boundary_head is None
    image = torch.randn(1, 3, 64, 96)
    with torch.no_grad():
        logits = boundary(image)['logits']
        inference = boundary.inference_model()
        assert all(not module.training for module in inference.modules())
        removed = inference(image)
    assert logits.shape == (1, 3, 64, 96)
    torch.testing.assert_close(logits, removed, rtol=0, atol=0)


def test_fp16_large_reductions_remain_finite():
    logits = torch.randn(4, 3, 384, 384, dtype=torch.float16, requires_grad=True)
    target = torch.randint(0, 3, (4, 384, 384))
    loss = segmentation_loss(logits, target) + distillation_loss(logits, logits.detach() + 0.1, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_real_segformer_frozen_gradients():
    teacher = Teacher('b0', False).eval()
    teacher.requires_grad_(False)
    student = Student(16, False, False).eval()
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        target = teacher(image)['logits']
    output = student(image)['logits']
    loss = distillation_loss(output, target, torch.zeros(2, 64, 64, dtype=torch.long))
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in student.parameters())


def test_audit_pairing_leakage_and_unknown_values(tmp_path):
    root = tmp_path / 'data'
    synthetic(root, 4)
    report = audit(root, tmp_path / 'audit', 'indexed', False)
    assert report['counts'] == {'train': 4, 'validation': 4, 'test': 4}
    data = SegmentationData(root, tmp_path / 'audit/train.json', 64, True)
    a, b = data[0]
    assert a.shape == (3, 64, 64) and b.shape == (64, 64)
    prepare_cache(root, tmp_path / 'audit/train.json', tmp_path / 'cache', 64)
    cached = SegmentationData(root, tmp_path / 'audit/train.json', 64, True, cache=tmp_path / 'cache')
    cached_a, cached_b = cached[0]
    torch.testing.assert_close(a, cached_a, rtol=0, atol=0)
    torch.testing.assert_close(b, cached_b, rtol=0, atol=0)
    image = root / 'train/images/smoke_000.png'
    (root / 'validation/images/smoke_000.png').write_bytes(image.read_bytes())
    mask = root / 'train/masks/smoke_000_mask.png'
    (root / 'validation/masks/smoke_000_mask.png').write_bytes(mask.read_bytes())
    with pytest.raises(ValueError, match='audit failed'):
        audit(root, tmp_path / 'bad_audit', 'indexed', False)
    assert any('Cross-split' in error for error in json.loads((tmp_path / 'bad_audit/audit.json').read_text())['errors'])


def test_train_resume_equivalence_and_config_guard(tmp_path):
    root, manifests = tmp_path / 'data', tmp_path / 'manifests'
    synthetic(root, 4)
    audit(root, manifests, 'indexed', False)
    config = smoke_config()
    train(config, root, manifests, tmp_path / 'full', 'cpu')
    partial = train(config, root, manifests, tmp_path / 'resumed', 'cpu', stop_after_epoch=1)
    assert partial['status'] == 'interrupted_at_epoch_boundary'
    continued = train(config, root, manifests, tmp_path / 'resumed', 'cpu', resume=True)
    assert continued['status'] == 'completed'
    first, second = load_checkpoint(tmp_path / 'full/last.pt'), load_checkpoint(tmp_path / 'resumed/last.pt')
    assert first['global_step'] == second['global_step'] == 4
    for name, value in first['model'].items():
        torch.testing.assert_close(value, second['model'][name], rtol=0, atol=0)
    changed = dict(config, lambda_boundary=0.8)
    with pytest.raises(ValueError, match='changed'):
        train(changed, root, manifests, tmp_path / 'full', 'cpu')


def test_onnx_alignment_and_headless_image_video(tmp_path):
    config = smoke_config()
    seed_everything(42)
    model = Student(16, True, False).eval()
    checkpoint = tmp_path / 'smoke.pt'
    torch.save({'config': config, 'model': model.state_dict(), 'smoke': True}, checkpoint)
    out = tmp_path / 'student_smoke.onnx'
    record = export_model(checkpoint, out, smoke=True)
    assert record['auxiliary_removed_exact']
    assert min(c['mask_agreement'] for c in record['comparisons']) >= 0.999
    image = np.zeros((48, 80, 3), np.uint8)
    image[:, 40:] = [25, 255, 40]
    write_image(tmp_path / 'sample.png', image)
    runner = ONNXRunner(out)
    try:
        result = demo(runner, tmp_path / 'sample.png', tmp_path / 'images', smoke=True, warmup=1)
        assert result['frames'] == 1 and result['smoke']
        video = tmp_path / 'input.avi'
        writer = cv2.VideoWriter(video_path(video), cv2.VideoWriter_fourcc(*'MJPG'), 10, (80, 48))
        assert writer.isOpened()
        for _ in range(8):
            writer.write(image)
        writer.release()
        result = demo(runner, video, tmp_path / 'video', smoke=True, warmup=1)
        assert result['frames'] == 8 and result['inference']['mean_ms'] > 0
        assert (tmp_path / 'video/demo.avi').stat().st_size > 0
        async_result = demo(runner, video, tmp_path / 'async', smoke=True, warmup=1, asynchronous=True)
        assert async_result['frames'] + async_result['dropped'] == 8
        with pytest.raises(RuntimeError, match='Cannot open'):
            demo(runner, tmp_path / 'missing.avi', tmp_path / 'missing', warmup=0)
    finally:
        runner.close()


def test_seeded_robustness_and_nano_syntax():
    image = np.full((16, 16, 3), 200, np.uint8)
    for condition in ('brightness', 'blur', 'jpeg', 'clean'):
        np.testing.assert_array_equal(corrupt(image, condition, 4), corrupt(image, condition, 4))
    for name in ('runtime.py', 'nano.py'):
        path = Path(name)
        if path.exists():
            ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 6))
