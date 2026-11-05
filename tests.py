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

from data import load_mask, synthetic, audit, SegmentationData, prepare_cache, sha256, prepare_splits, cache_original_targets, pack_labels, unpack_labels
from learning import seed_everything, train, load_checkpoint, export_model, record_progress
from metrics import Metrics, corrupt, tensor_statistics
from models import Student, Teacher, segmentation_loss, boundary_targets, boundary_loss, distillation_loss
from runtime import letterbox, preprocess, restore_logits, ONNXRunner, demo, video_path, write_image, read_image

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


def test_packed_targets_preserve_every_label_and_odd_shapes():
    target = np.array([[0, 1, 2, 255, 1], [255, 0, 2, 1, 0], [2, 255, 1, 0, 2]], np.uint8)
    packed = pack_labels(target)
    assert len(packed) == 4
    np.testing.assert_array_equal(unpack_labels(packed, target.shape), target)
    with pytest.raises(ValueError, match='Unknown'):
        pack_labels(np.array([[3]], np.uint8))


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


@pytest.mark.parametrize('device', ['cpu'] + (['cuda'] if torch.cuda.is_available() else []))
def test_tensor_metrics_match_reference_exactly(device):
    rng = np.random.default_rng(7)
    for tolerance in (0, 2):
        for shape in ((7, 13), (101, 211)):
            target = rng.integers(0, 3, shape, dtype=np.uint8)
            predicted = rng.integers(0, 3, shape, dtype=np.uint8)
            target[::4, ::3] = 255
            predicted[target == 255] = 99
            reference = Metrics(tolerance)
            reference.update(predicted, target)
            counts, boundary = tensor_statistics(torch.from_numpy(predicted).to(device), torch.from_numpy(target).to(device), tolerance)
            np.testing.assert_array_equal(counts, reference.confusion)
            np.testing.assert_array_equal(boundary, reference.boundary)
    for value in (0, 255):
        target = np.full((9, 14), value, np.uint8)
        predicted = np.zeros_like(target)
        reference = Metrics()
        reference.update(predicted, target)
        counts, boundary = tensor_statistics(torch.from_numpy(predicted).to(device), torch.from_numpy(target).to(device))
        np.testing.assert_array_equal(counts, reference.confusion)
        np.testing.assert_array_equal(boundary, reference.boundary)


def test_image_loading_uses_same_raw_grid_despite_exif(tmp_path):
    pixels = np.zeros((20, 40, 3), np.uint8)
    pixels[:, :20] = [230, 25, 50]
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / 'orientation.jpg'
    Image.fromarray(pixels).save(path, exif=exif, quality=95)
    expected = np.asarray(Image.open(path).convert('RGB'))
    actual = cv2.cvtColor(read_image(path), cv2.COLOR_BGR2RGB)
    assert actual.shape == expected.shape == (20, 40, 3)
    np.testing.assert_allclose(actual, expected, atol=2)


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
    prepare_cache(root, tmp_path / 'audit/train.json', tmp_path / 'eval_cache', 64, ['clean', 'brightness', 'blur', 'jpeg'])
    eval_cached = SegmentationData(root, tmp_path / 'audit/train.json', 64, cache=tmp_path / 'eval_cache')
    for condition in ('clean', 'brightness', 'blur', 'jpeg'):
        expected, expected_meta, expected_target = data.evaluation_item(0, condition)
        actual, actual_meta, actual_target = eval_cached.evaluation_item(0, condition)
        np.testing.assert_array_equal(expected, actual)
        np.testing.assert_array_equal(expected_target, actual_target)
        assert expected_meta == actual_meta
    cache_original_targets(root, tmp_path / 'audit/train.json', tmp_path / 'eval_cache')
    target_cached = SegmentationData(root, tmp_path / 'audit/train.json', 64, cache=tmp_path / 'eval_cache')
    for i in range(len(data)):
        np.testing.assert_array_equal(target_cached.target(i), data.target(i))
    image = root / 'train/images/smoke_000.png'
    (root / 'validation/images/smoke_000.png').write_bytes(image.read_bytes())
    mask = root / 'train/masks/smoke_000_mask.png'
    (root / 'validation/masks/smoke_000_mask.png').write_bytes(mask.read_bytes())
    with pytest.raises(ValueError, match='audit failed'):
        audit(root, tmp_path / 'bad_audit', 'indexed', False)
    assert any('Cross-split' in error for error in json.loads((tmp_path / 'bad_audit/audit.json').read_text())['errors'])
    cleaned = prepare_splits(tmp_path / 'bad_audit', tmp_path / 'derived')
    assert cleaned['counts'] == {'train': 3, 'validation': 4, 'test': 4}
    assert cleaned['excluded']['train'] == ['train/images/smoke_000.png']
    assert cleaned['test_preserved']
    bad = json.loads((tmp_path / 'bad_audit/audit.json').read_text())
    bad['errors'].append('Unknown labels')
    (tmp_path / 'bad_audit/audit.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='non-duplicate'):
        prepare_splits(tmp_path / 'bad_audit', tmp_path / 'refused')


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


def test_geometry_recipe_and_audit_journal_recovery(tmp_path):
    root, destination = tmp_path / 'data', tmp_path / 'audit'
    synthetic(root, 4)
    mask_path = root / 'train/masks/smoke_000_mask.png'
    original = np.asarray(Image.open(mask_path)).copy()
    Image.fromarray(np.rot90(original, 1)).save(mask_path)
    with pytest.raises(ValueError, match='audit failed'):
        audit(root, destination, 'indexed', False)
    recipe = {'corrections': {'train/images/smoke_000.png': {
        'mask_rotation_cw': 90, 'reviewed': True,
        'image_sha256': sha256(root / 'train/images/smoke_000.png'), 'mask_sha256': sha256(mask_path)}}}
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps(recipe))
    report = audit(root, destination, 'indexed', False, path)
    assert report['passed'] and report['journal_reused'] == 11
    dataset = SegmentationData(root, destination / 'train.json', 64)
    np.testing.assert_array_equal(dataset.original(0)[1], original)
    journal = destination / 'audit_rows.jsonl'
    with journal.open('ab') as handle:
        handle.write(b'{"image": "interrupted')
    assert audit(root, destination, 'indexed', False, path)['passed']
    assert audit(root, destination, 'indexed', False, path)['journal_reused'] == 12
    recipe['corrections']['train/images/smoke_000.png']['mask_sha256'] = '0' * 64
    path.write_text(json.dumps(recipe))
    with pytest.raises(ValueError, match='audit failed'):
        audit(root, destination, 'indexed', False, path)


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


def test_summary_refuses_mixed_dataset_identities(tmp_path):
    from report import summarize
    for name, fingerprint in [('S0', 'dataset-a'), ('S1', 'dataset-b')]:
        folder = tmp_path / name
        folder.mkdir()
        run = {'status': 'completed', 'identity': name, 'smoke': False, 'config': smoke_config(),
               'provenance': {'sources': {'models.py': 'same'}, 'manifests': {'train.json': fingerprint}}}
        (folder / 'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='experimental protocol'):
        summarize(tmp_path, tmp_path / 'summary')
    assert not (tmp_path / 'summary/results.json').exists()


def test_progress_updates_preserve_manual_notes(tmp_path):
    document = tmp_path / 'PROGRESS.md'
    document.write_text('Manual note: keep the original data.\n', encoding='utf-8')
    root = tmp_path / 'runs'
    (root / 'T').mkdir(parents=True)
    (root / 'T/run.json').write_text(json.dumps({'status': 'running', 'last_completed_epoch': 1,
                                               'config': {'epochs': 20}, 'best_miou': 0.5}))
    record_progress(root, 'training', document)
    record_progress(root, 'validation', document)
    text = document.read_text(encoding='utf-8')
    assert text.count('CV_CW2_AUTO_PROGRESS_BEGIN') == 1
    assert 'Manual note: keep the original data.' in text
    assert 'validation' in text and '1/20' in text


def test_packaging_refuses_incomplete_experiments(tmp_path):
    from package_artifacts import package
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'pipeline.json').write_text(json.dumps({'status': 'running', 'experiments': ['T']}))
    with pytest.raises(ValueError, match='must finish'):
        package(runs, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()
    (runs / 'pipeline.json').write_text(json.dumps({'status': 'completed', 'experiments': ['T']}))
    with pytest.raises(ValueError, match='All five'):
        package(runs, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def test_evaluation_reuse_rejects_partial_or_stale_results(tmp_path):
    from project import evaluation_is_current
    path = tmp_path / 'evaluation.json'
    result = {'run_identity': 'run', 'checkpoint_sha256': 'weights', 'manifest_sha256': 'data',
              'corruption': 'clean', 'samples': 2, 'per_image': [{}, {}], 'smoke': False}
    path.write_text(json.dumps(result))
    assert evaluation_is_current(path, 'run', 'weights', 'data', 'clean', 2)
    assert not evaluation_is_current(path, 'run', 'other-weights', 'data', 'clean', 2)
    assert not evaluation_is_current(path, 'run', 'weights', 'other-data', 'clean', 2)
    assert not evaluation_is_current(path, 'run', 'weights', 'data', 'blur', 2)
    assert not evaluation_is_current(path, 'run', 'weights', 'data', 'clean', 3)
    result['per_image'] = [{}]
    path.write_text(json.dumps(result))
    assert not evaluation_is_current(path, 'run', 'weights', 'data', 'clean', 2)


def test_figures_use_full_test_scores_without_cherry_picking(tmp_path):
    from project import prepare_figure_manifest
    manifest_path, evaluation_path = tmp_path / 'manifest.json', tmp_path / 'evaluation.json'
    manifest = {'records': [{'image': name} for name in ('a', 'b', 'c', 'd', 'e')], 'audit_passed': True}
    manifest_path.write_text(json.dumps(manifest))
    evaluation = {'manifest_sha256': sha256(manifest_path), 'checkpoint_sha256': 'model', 'corruption': 'clean',
                  'smoke': False, 'samples': 5, 'per_image': [{'image': name, 'miou': score} for name, score in
                    [('e', 0.2), ('c', 0.1), ('b', 0.8), ('a', 0.9), ('d', None)]]}
    evaluation_path.write_text(json.dumps(evaluation))
    indices = prepare_figure_manifest(manifest_path, evaluation_path, tmp_path / 'figures.json', 'model', limit=2)
    assert indices == [0, 1, 2, 4]
    assert json.loads((tmp_path / 'figures.json').read_text())['split'] == 'figures_only'


def test_nano_report_refuses_unverified_hardware(tmp_path):
    from report import summarize_nano
    path = tmp_path / 'unverified.json'
    path.write_text(json.dumps({'S3_384_fp16': {'environment': {'target_matches': False}}}))
    with pytest.raises(ValueError, match='actual original-Nano'):
        summarize_nano([path], tmp_path / 'report')
    assert not (tmp_path / 'report/nano_results.json').exists()


def test_release_bundle_includes_manifests_and_rejects_tampering(tmp_path):
    import hashlib
    import zipfile
    from publish_artifacts import validate_bundle
    names = {'nano.py', 'runtime.py', 'selection.json', 'ARTIFACTS.md'}
    names.update('demo/' + name for name in ('input_replay.mp4', 'S0.mp4', 'final.mp4', 'S0_timings.json', 'final_timings.json', 'provenance.json'))
    names.update('manifests/' + name for name in ('train.json', 'validation.json', 'test.json', 'audit.json'))
    result_files = ('run.json', 'epochs.jsonl', 'best_validation.json', 'test_clean.json',
                    'test_brightness.json', 'test_blur.json', 'test_jpeg.json')
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        names.add('checkpoints/' + name + '.pt')
        names.update('results/' + name + '/' + item for item in result_files)
    for name, size in (('S3', 256), ('S3', 384), ('S3', 512), ('S0', 384)):
        names.update('models/{}_{}{}'.format(name, size, suffix) for suffix in ('.onnx', '.json', '.reference.npz'))
    payload = b'unit-test archive content; no model weights'
    manifest = {'source_commit': 'a' * 40, 'selected': 'S3',
                'files': [{'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
                          for name in sorted(names)]}
    archive = tmp_path / 'fixture.zip'

    def make_archive(tampered=False, extra=False):
        with zipfile.ZipFile(archive, 'w') as stream:
            stream.writestr('manifest.json', json.dumps(manifest))
            for name in sorted(names):
                stream.writestr(name, b'changed' if tampered and name == 'manifests/train.json' else payload)
            if extra:
                stream.writestr('unrelated.txt', b'unrelated')
        recipe = {'source_commit': 'a' * 40, 'selected': 'S3', 'archive': archive.name,
                  'bytes': archive.stat().st_size, 'sha256': sha256(archive)}
        archive.with_suffix('.release.json').write_text(json.dumps(recipe))
        return recipe

    recipe = make_archive()
    assert validate_bundle(archive) == recipe
    make_archive(tampered=True)
    with pytest.raises(ValueError, match='checksum mismatch'):
        validate_bundle(archive)
    make_archive(extra=True)
    with pytest.raises(ValueError, match='Unexpected archive entries'):
        validate_bundle(archive)


def test_precision_configuration_and_mixed_report_refusal(tmp_path):
    from learning import amp_settings
    from report import summarize
    assert amp_settings({'amp': True, 'amp_dtype': 'bfloat16'}, 'cpu') == (False, torch.bfloat16, False)
    with pytest.raises(ValueError, match='amp_dtype'):
        amp_settings({'amp': True, 'amp_dtype': 'unknown'}, 'cpu')
    for name, dtype in [('S0', 'float16'), ('S1', 'bfloat16')]:
        folder = tmp_path / name
        folder.mkdir()
        config = dict(smoke_config(), amp_dtype=dtype)
        run = {'status': 'completed', 'identity': name, 'smoke': False, 'config': config,
               'provenance': {'sources': {'models.py': 'same'}, 'manifests': {'train.json': 'same'}}}
        (folder / 'run.json').write_text(json.dumps(run))
    with pytest.raises(ValueError, match='experimental protocol'):
        summarize(tmp_path, tmp_path / 'report')


def test_teacher_reuse_preserves_identity_and_refuses_changed_data(tmp_path):
    import hashlib
    from project import reuse_teacher
    source, target, manifests = [tmp_path / name for name in ('source', 'target', 'manifests')]
    source.mkdir()
    manifests.mkdir()
    config = {'kind': 'teacher', 'epochs': 1}
    for split in ('train', 'validation'):
        (manifests / (split + '.json')).write_text(json.dumps({'split': split}))
    hashes = {p.name: sha256(p) for p in manifests.iterdir()}
    provenance = {'config': config, 'sources': {'learning.py': 'original'}, 'manifests': hashes}
    identity = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    run = {'status': 'completed', 'smoke': False, 'config': config, 'provenance': provenance,
           'identity': identity, 'last_completed_epoch': 1, 'git_commit_at_start': 'a' * 40}
    (source / 'run.json').write_text(json.dumps(run))
    (source / 'epochs.jsonl').write_text(json.dumps({'epoch': 1}) + '\n')
    validation = {'miou': 0.8}
    (source / 'best_validation.json').write_text(json.dumps(validation))
    for name in ('best', 'last'):
        torch.save({'identity': identity, 'config': config, 'smoke': False, 'epoch': 1,
                    'validation_manifest_sha256': hashes['validation.json'], 'validation': validation}, source / (name + '.pt'))
    assert reuse_teacher(source, target, manifests, config) == run
    assert reuse_teacher(source, target, manifests, config) == run
    assert sha256(source / 'run.json') == sha256(target / 'run.json')
    (manifests / 'train.json').write_text('changed')
    with pytest.raises(ValueError, match='data manifests differ'):
        reuse_teacher(source, target, manifests, config)
    assert json.loads((target / 'run.json').read_text()) == run


def test_delivery_refuses_partial_pipeline_and_replaces_only_its_section(tmp_path):
    from delivery import verify_results, replace_section
    runs = tmp_path / 'runs'
    runs.mkdir()
    (runs / 'pipeline.json').write_text(json.dumps({'status': 'failed', 'experiments': ['T']}))
    with pytest.raises(ValueError, match='All five complete'):
        verify_results(runs, tmp_path / 'manifests')
    note = tmp_path / 'notes.md'
    note.write_text('Manual evidence\n', encoding='utf-8')
    replace_section(note, 'first', prepend=True)
    replace_section(note, 'second')
    text = note.read_text(encoding='utf-8')
    assert text.count('ACTUAL_DELIVERY_BEGIN') == 1
    assert text.startswith('<!-- ACTUAL_DELIVERY_BEGIN -->')
    assert 'Manual evidence' in text and 'second' in text and 'first' not in text
