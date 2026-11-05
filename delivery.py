import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from filelock import FileLock
from PIL import Image

from data import sha256
from learning import identity, write_json
from project import config_file, evaluation_is_current, export_is_current


def run_command(arguments, cwd=None):
    return subprocess.run([str(x) for x in arguments], cwd=cwd, check=True, capture_output=True, text=True, encoding='utf-8')


def verify_results(runs, manifests):
    runs, manifests = Path(runs), Path(manifests)
    pipeline = config_file(runs / 'pipeline.json')
    if pipeline.get('status') != 'completed' or set(pipeline.get('experiments', [])) != {'T', 'S0', 'S1', 'S2', 'S3'}:
        raise ValueError('All five complete experiments are required for delivery')
    test_path = manifests / 'test.json'
    test = config_file(test_path)
    expected_images = {row['image'] for row in test['records']}
    if len(expected_images) != len(test['records']):
        raise ValueError('Test manifest contains repeated paths')
    scores, hashes = [], {}
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        run = config_file(runs / name / 'run.json')
        if run['status'] != 'completed' or run['smoke'] or run['last_completed_epoch'] != run['config']['epochs']:
            raise ValueError('Incomplete formal run: ' + name)
        if name != 'T':
            current, _ = identity(run['config'], [manifests / 'train.json', manifests / 'validation.json'])
            if current != run['identity']:
                raise ValueError('Student training source/config/data differs: ' + name)
        validation = config_file(runs / name / 'best_validation.json')
        if (validation['manifest_sha256'] != sha256(manifests / 'validation.json') or
                validation['samples'] != len(config_file(manifests / 'validation.json')['records'])):
            raise ValueError('Incomplete or stale validation: ' + name)
        if name != 'T':
            scores.append((validation['miou'], name))
        hashes[name] = sha256(runs / name / 'best.pt')
        for condition in ('clean', 'brightness', 'blur', 'jpeg'):
            path = runs / name / ('test_' + condition + '.json')
            if not evaluation_is_current(path, run['identity'], hashes[name], sha256(test_path), condition, len(expected_images)):
                raise ValueError('Incomplete or stale test: ' + name + '/' + condition)
            if {row['image'] for row in config_file(path)['per_image']} != expected_images:
                raise ValueError('Test results do not cover the exact manifest')
    selected = config_file(runs / 'selection.json')['selected']
    if selected != max(scores)[1]:
        raise ValueError('Selected student differs from validation-only selection')
    for name, size in {(selected, size) for size in (256, 384, 512)} | {('S0', 384)}:
        if not export_is_current(runs / 'exports' / ('{}_{}.onnx'.format(name, size)), hashes[name], size):
            raise ValueError('Export provenance/alignment receipt differs')
    return selected


def make_replay(runs, selected, video, sources, output, ffmpeg):
    from runtime import ONNXRunner, demo
    runs, output, video = Path(runs), Path(output), Path(video)
    output.mkdir(parents=True, exist_ok=True)
    source_record = config_file(sources)
    if source_record.get('source_type') != 'validation still-image replay, not live footage':
        raise ValueError('Expected the documented validation still-image replay')
    records = source_record['records']
    if not records or not all(row['image'].startswith('validation/') for row in records):
        raise ValueError('Replay contains undocumented source images')
    expected_frames = sum(row['frames'] for row in records)
    checkpoint_hashes = {name: sha256(runs / name / 'best.pt') for name in {'S0', selected}}
    provenance_path = output / 'provenance.json'
    if provenance_path.exists():
        previous = config_file(provenance_path)
        if (previous['selected'] == selected and previous['checkpoints'] == checkpoint_hashes and
                previous['input_sha256'] == sha256(video) and
                all(sha256(output / name) == digest for name, digest in previous['files'].items())):
            return previous
        raise ValueError('Existing replay differs; choose a new delivery directory')

    def encode(source, destination):
        run_command([ffmpeg, '-nostdin', '-y', '-loglevel', 'error', '-i', source, '-an',
                     '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', destination])

    encode(video, output / 'input_replay.mp4')
    for label, name in (('S0', 'S0'), ('final', selected)):
        model = runs / 'exports' / (name + '_384.onnx')
        runner = ONNXRunner(model)
        try:
            timing = demo(runner, video, output / (label + '_raw'), headless=True,
                          label='REPLAY VALIDATION IMAGES ' + name, smoke=False)
        finally:
            runner.close()
        if timing['frames'] != expected_frames or timing['source_type'] != 'playback' or timing['smoke']:
            raise ValueError('Replay frame count or source type differs')
        encode(output / (label + '_raw/demo.avi'), output / (label + '.mp4'))
        write_json(output / (label + '_timings.json'), timing)
    names = ('input_replay.mp4', 'S0.mp4', 'final.mp4', 'S0_timings.json', 'final_timings.json')
    provenance = {'source_type': 'validation_still_image_replay', 'source': source_record,
                  'selected': selected, 'checkpoints': checkpoint_hashes, 'input_sha256': sha256(video),
                  'frames': expected_frames, 'nano_measurement': False,
                  'files': {name: sha256(output / name) for name in names}}
    write_json(provenance_path, provenance)
    return provenance


def replace_section(path, text, prepend=False):
    path = Path(path)
    begin, end = '<!-- ACTUAL_DELIVERY_BEGIN -->', '<!-- ACTUAL_DELIVERY_END -->'
    with FileLock(str(path) + '.lock', timeout=10):
        original = path.read_text(encoding='utf-8') if path.exists() else ''
        section = begin + '\n' + text.strip() + '\n' + end
        if begin in original:
            before, _, rest = original.partition(begin)
            _, _, after = rest.partition(end)
            text = before + section + after
        elif prepend:
            text = section + '\n\n' + original
        else:
            text = original.rstrip() + '\n\n' + section + '\n'
        path.write_text(text, encoding='utf-8')


def prepare_public_results(runs, selected):
    from report import summarize
    output = Path('results/final')
    rows = summarize(runs, output)
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        for filename in ('run.json', 'epochs.jsonl'):
            shutil.copy2(runs / name / filename, output / (name + '_' + filename))
    comparison = config_file(runs / 'comparison/comparison_receipt.json')
    for entry in comparison['files']:
        source = runs / 'comparison' / entry['path']
        if sha256(source) != entry['sha256']:
            raise ValueError('Comparison figure content differs')
        if source.suffix == '.png':
            with Image.open(source) as original:
                preview = original.convert('RGB')
                preview.thumbnail((2400, 1200))
                preview.save(output / (source.stem + '.jpg'), quality=90)
        elif source.name == 'selection.json':
            shutil.copy2(source, output / 'figure_selection.json')
    shutil.copy2(runs / 'calibration/selection.json', output / 'calibration.json')
    write_json(output / 'deployment_selection.json', config_file(runs / 'selection.json'))
    lines = ['## Completed host experiments', '',
             'Protocol: Trans10K v1 leak-free-v1, 4990/999/4428. All models completed 20 epochs. All students used BF16 AMP.',
             'The frozen teacher retains its original FP16 training provenance. Deployment selection used validation only: ' + selected + '.', '']
    for name, row in rows.items():
        clean = row['clean']
        lines.append('{}: validation mIoU {:.6f}; test mIoU {:.6f}; foreground IoU {:.6f}; foreground Dice {:.6f}; boundary F1 {:.6f}.'.format(
            name, row['validation_miou'], clean['miou'], clean['foreground_iou'], clean['foreground_dice'], clean['boundary_f1']))
        lines.append('Class IoUs: {}. Brightness/blur/JPEG mIoU: {:.6f}/{:.6f}/{:.6f}.'.format(
            clean['per_class_iou'], row['brightness']['miou'], row['blur']['miou'], row['jpeg']['miou']))
        lines.append('')
    for name in ('S1', 'S2', 'S3'):
        lines.append('{} versus S0: clean-test mIoU change {:+.4f} percentage points; boundary-F1 change {:+.4f} percentage points.'.format(
            name, 100 * (rows[name]['clean']['miou'] - rows['S0']['clean']['miou']),
            100 * (rows[name]['clean']['boundary_f1'] - rows['S0']['clean']['boundary_f1'])))
    lines += ['', 'Negative changes are retained. This single-seed comparison does not establish statistical significance.',
              'Figures in results/final are generated from actual logs. The four image columns are original / ground truth / S0 / selected student.',
              'JPEG figure copies are display-sized previews; all metrics use original-resolution labels.',
              'Nano latency, power, temperatures and live camera quality remain pending physical hardware.',
              '', '![Measured accuracy and robustness](../results/final/metrics.png)',
              '![Actual learning curves](../results/final/learning_curves.png)']
    replace_section('docs/REPORT_DRAFT.md', '\n'.join(lines))
    replace_section('docs/EXPERIMENTS.md', '\n'.join(lines[:4]) + '\n\nFull measured metrics: ../results/final/RESULTS.md.')
    return rows


def commit_and_push(paths, message):
    remote = run_command(['git', 'remote', 'get-url', 'origin']).stdout.strip().rstrip('/')
    if remote not in ('https://github.com/hogomex393-svg/Computer-Vision-cw2.git', 'https://github.com/hogomex393-svg/Computer-Vision-cw2'):
        raise ValueError('Unexpected GitHub destination')
    branch = run_command(['git', 'branch', '--show-current']).stdout.strip()
    if branch != 'codex/transparent-segmentation':
        raise ValueError('Unexpected delivery branch')
    if run_command(['git', 'diff', '--cached', '--name-only']).stdout.strip():
        raise ValueError('Unrelated staged edits exist; do not include them in delivery')
    run_command(['git', 'add', '--', *paths])
    if run_command(['git', 'diff', '--cached', '--name-only']).stdout.strip():
        run_command(['git', 'commit', '-m', message])
    run_command(['git', 'push', 'origin', branch])
    local = run_command(['git', 'rev-parse', 'HEAD']).stdout.strip()
    remote_sha = run_command(['git', 'ls-remote', 'origin', 'refs/heads/' + branch]).stdout.split()[0]
    if local != remote_sha:
        raise RuntimeError('Remote branch does not match the delivered source')
    return local


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--manifests', required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--video-sources', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--cpu-python', default=sys.executable)
    parser.add_argument('--out', required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--account', default='DzAsy11')
    parser.add_argument('--tag', default='cw2-host-bf16-v1')
    args = parser.parse_args()
    runs, output = Path(args.runs), Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'delivery_state.json'

    def stage(name, **values):
        write_json(state_path, dict(stage=name, **values))
        print(json.dumps(dict(stage=name, **values), ensure_ascii=False), flush=True)

    with FileLock(str(output / '.delivery.lock'), timeout=0):
        try:
            if args.wait:
                stage('waiting_for_pipeline')
                while not (runs / 'pipeline.json').exists():
                    time.sleep(30)
                if config_file(runs / 'pipeline.json')['status'] != 'completed':
                    raise RuntimeError('Training pipeline failed; inspect its diagnostic logs')
            stage('validating_all_real_results')
            selected = verify_results(runs, args.manifests)
            stage('generating_validation_replay', selected=selected)
            make_replay(runs, selected, args.video, args.video_sources, output / 'demo', args.ffmpeg)
            stage('generating_reports', selected=selected)
            prepare_public_results(runs, selected)
            if not args.upload:
                stage('prepared_for_review', selected=selected)
                return
            stage('committing_verified_results')
            source_commit = commit_and_push(['results/final', 'docs/REPORT_DRAFT.md', 'docs/EXPERIMENTS.md'],
                                            'results: record complete real-data experiments and figures')
            clone = output / ('remote_check_' + source_commit[:12])
            if not clone.exists():
                run_command(['git', 'clone', '--depth', '1', '--branch', 'codex/transparent-segmentation',
                             'https://github.com/hogomex393-svg/Computer-Vision-cw2.git', clone])
            if run_command(['git', 'rev-parse', 'HEAD'], cwd=clone).stdout.strip() != source_commit:
                raise ValueError('Remote check clone has a different commit')
            stage('validating_remote_checkout', source_commit=source_commit)
            test = run_command([args.cpu_python, '-m', 'pytest', 'tests.py', '-q', '--basetemp', 'artifacts/delivery_tests',
                                '--junitxml', 'artifacts/delivery_pytest.xml'], cwd=clone)
            (output / 'remote_pytest.txt').write_text(test.stdout, encoding='utf-8')
            stage('packaging_models_and_replay')
            from package_artifacts import package
            from publish_artifacts import validate_bundle, publish
            bundle = output / 'cw2_models_bf16_v1'
            archive = bundle.with_suffix('.zip')
            if not archive.exists():
                package(runs, bundle, args.manifests, output / 'demo')
            recipe = validate_bundle(archive)
            stage('uploading_collaborator_only_draft', bytes=recipe['bytes'], sha256=recipe['sha256'])
            uploaded = publish(archive, args.account, args.tag)
            write_json('results/final/delivery.json', uploaded)
            message = ('## 主机实验与模型交付已完成\n\n'
                       'T、S0、S1、S2、S3 均完成 20 轮；全部模型完成 4428 张原图上的四种条件测试。'
                       '最终学生按验证集选择为 ' + selected + '，正式模型和同一输入的 S0/最终模型演示已打包。\n\n'
                       '模型包：[' + archive.name + '](' + uploaded['release_url'] + ')。'
                       '需要有仓库推送权限的 GitHub 账号登录；保持 Release 为草稿，不点击 Publish。\n\n'
                       '体积：' + str(uploaded['bytes']) + ' 字节。SHA256：`' + uploaded['sha256'] + '`。'
                       '包内源码版本：`' + uploaded['source_commit'] + '`。\n\n'
                       '模型包含五个最佳 checkpoint、最终学生 256/384/512 ONNX、S0 384 ONNX、对齐参考、实验记录、数据清单和验证静态图回放。'
                       '已实际验证 GitHub 上传摘要、匿名访问拒绝和远端检出的 CPU 回归。\n\n'
                       '仍需组员：按 JETSON.md 在原版 Nano 构建引擎、对齐和十分钟性能矩阵；按 DEMO_SCRIPT.md 做实拍。'
                       '本机回放不代表 Nano 实测。')
            for path in ('docs/HANDOFF.md', 'docs/PROGRESS.md', 'README.md'):
                replace_section(path, message, prepend=True)
            state = config_file('docs/work_state.json')
            state.update(status='complete', current_item_id=None,
                         completion_scope='Host L3 training, evaluation, ONNX, replay and verified GitHub handoff; original Nano hardware acceptance remains external')
            for item in state['items']:
                item['status'] = 'completed'
            state['gates'].update(disabled_release='passed', real_acceptance='passed')
            state['external_acceptance'] = {'nano_hardware': 'pending_no_connection', 'live_filming': 'pending_team_capture'}
            state['checkpoints'].append({'revision': max(c['revision'] for c in state['checkpoints']) + 1,
                'item_id': 'W04', 'state': 'Complete host L3 delivery; target Nano is explicitly unverified',
                'evidence': ['results/final/results.json', 'results/final/delivery.json', str(output / 'remote_pytest.txt'),
                             uploaded['release_url'], uploaded['sha256']],
                'next_action': 'Team executes original-Nano validation and live filming using the documented commands',
                'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')})
            write_json('docs/work_state.json', state)
            stage('committing_final_handoff')
            final_commit = commit_and_push(['results/final/delivery.json', 'docs/HANDOFF.md', 'docs/PROGRESS.md', 'docs/work_state.json', 'README.md'],
                                           'docs: publish verified collaborator model handoff')
            stage('completed', selected=selected, source_commit=uploaded['source_commit'], final_commit=final_commit,
                  release_url=uploaded['release_url'], sha256=uploaded['sha256'], nano_hardware_verified=False)
        except Exception as error:
            if state_path.exists() and config_file(state_path).get('stage') == 'committing_final_handoff':
                state = config_file('docs/work_state.json')
                state.update(status='active', current_item_id='W04')
                for item in state['items']:
                    if item['item_id'] == 'W04':
                        item['status'] = 'in_progress'
                state['checkpoints'].append({'revision': max(c['revision'] for c in state['checkpoints']) + 1,
                    'item_id': 'W04', 'state': 'Model assets verified; final handoff documentation push failed',
                    'evidence': [str(state_path)], 'next_action': 'Resolve the recorded Git failure and finish normal push',
                    'created_at': datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')})
                write_json('docs/work_state.json', state)
            stage('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
