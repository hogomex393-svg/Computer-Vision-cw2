import argparse
import json
from pathlib import Path


def summarize(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, fingerprints, protocols = {}, set(), set()
    dataset_identities, test_identities, data_protocols = set(), set(), set()
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        directory = root / name
        if not (directory / 'run.json').exists():
            rows[name] = {'status': 'pending'}
            continue
        run = json.loads((directory / 'run.json').read_text())
        row = {'status': run['status'], 'identity': run['identity'], 'config': run['config'], 'smoke': run['smoke']}
        if run['smoke']:
            raise ValueError('Formal result summary refuses smoke runs')
        dataset_identities.add(json.dumps(run['provenance']['manifests'], sort_keys=True))
        if name != 'T':
            fingerprints.add(json.dumps(run['provenance']['sources'], sort_keys=True))
            protocol = {key: run['config'][key] for key in (
                'size', 'batch_size', 'epochs', 'seed', 'augment', 'learning_rate', 'weight_decay', 'freeze_bn', 'decoder_channels', 'amp')}
            protocol['amp_dtype'] = run['config'].get('amp_dtype', 'float16')
            protocols.add(json.dumps(protocol, sort_keys=True))
        for condition in ('clean', 'brightness', 'blur', 'jpeg'):
            path = directory / ('test_' + condition + '.json')
            if path.exists():
                evaluated = json.loads(path.read_text())
                if evaluated['run_identity'] != run['identity'] or evaluated['smoke']:
                    raise ValueError('Stale or smoke evaluation found: ' + str(path))
                test_identities.add(evaluated['manifest_sha256'])
                data_protocols.add(evaluated['data_protocol'])
                row[condition] = {k: v for k, v in evaluated.items() if k != 'per_image'}
        validation = directory / 'best_validation.json'
        if validation.exists():
            row['validation_miou'] = json.loads(validation.read_text())['miou']
        rows[name] = row
    if len(fingerprints) > 1 or len(protocols) > 1 or len(dataset_identities) > 1 or len(test_identities) > 1 or len(data_protocols) > 1:
        raise ValueError('Student runs do not share code and experimental protocol')
    (output / 'results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# Measured experiment results', '', 'Generated from run/evaluation files. Missing measurements remain pending.', '']
    if data_protocols:
        lines.extend(['Data protocol: ' + next(iter(data_protocols)) + '.', ''])
    available = []
    for name, row in rows.items():
        lines.append('## ' + name)
        lines.append('Status: ' + row['status'] + '.\n')
        if 'clean' in row:
            for metric in ('miou', 'per_class_iou', 'foreground_iou', 'foreground_dice', 'boundary_f1'):
                lines.append('- {}: {}'.format(metric, row['clean'][metric]))
            lines.append('- Identity: ' + row['identity'])
            available.append(name)
        lines.append('')
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    if available:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(available, [rows[n]['clean']['miou'] for n in available])
        protocol_label = next(iter(data_protocols)) if data_protocols else 'pending protocol'
        axes[0].set(ylabel='mIoU', ylim=(0, 1), title='Trans10K v1: ' + protocol_label)
        for condition in ('clean', 'brightness', 'blur', 'jpeg'):
            names = [n for n in available if condition in rows[n]]
            axes[1].plot(names, [rows[n][condition]['miou'] for n in names], marker='o', label=condition)
        axes[1].set(ylabel='mIoU', ylim=(0, 1), title='Fixed corruption protocol')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / 'metrics.png', dpi=180)
        plt.close(fig)
    histories = {}
    for name in ('T', 'S0', 'S1', 'S2', 'S3'):
        path = root / name / 'epochs.jsonl'
        if path.exists():
            histories[name] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if histories:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for name, history in histories.items():
            axes[0].plot([r['epoch'] for r in history], [r['loss'] for r in history], label=name)
            axes[1].plot([r['epoch'] for r in history], [r['validation_miou'] for r in history], label=name)
        axes[0].set(xlabel='Completed epoch', ylabel='Configured total loss', title='Loss terms differ across experiments')
        axes[1].set(xlabel='Completed epoch', ylabel='Validation mIoU', ylim=(0, 1), title='Actual logged validation curves')
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / 'learning_curves.png', dpi=180)
        plt.close(fig)
    return rows


def summarize_nano(paths, output):
    output = Path(output)
    rows = []
    for path in paths:
        matrix = json.loads(Path(path).read_text(encoding='utf-8'))
        for name, record in matrix.items():
            if (record.get('environment', {}).get('target_matches') is not True or record.get('smoke') is not False or
                    record.get('engine', {}).get('smoke') is not False or record.get('sustained_duration_met') is not True or
                    record.get('requested_duration_seconds', 0) < 600):
                raise ValueError('Only actual original-Nano non-smoke sustained measurements can enter the Nano report')
            rows.append({'case': name, 'matrix': Path(path).name, 'source_type': record['source_type'],
                         'precision': record['engine']['precision'], 'input_size': record['input_size'],
                         'inference': record['inference'], 'end_to_end': record['end_to_end'],
                         'processed_fps_wall': record['processed_fps_wall'], 'frame_age': record['frame_age'],
                         'dropped': record['dropped'], 'resources': record['resources'],
                         'environment': record['environment']})
    if not rows:
        raise ValueError('No Nano measurements supplied')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'nano_results.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    lines = ['# Real Nano measurements', '', 'Generated only from original-Nano sustained benchmark receipts.', '']
    for row in rows:
        lines.extend(['## ' + row['case'], '',
                      'Source: {}. Precision: {}. Input: {}.'.format(row['source_type'], row['precision'], row['input_size']),
                      'Processing FPS: {}. End-to-end mean/P50/P95 ms: {}/{}/{}.'.format(
                          row['processed_fps_wall'], row['end_to_end']['mean_ms'], row['end_to_end']['p50_ms'], row['end_to_end']['p95_ms']),
                      'Dropped frames: {}. Resources: {}.'.format(row['dropped'], json.dumps(row['resources'])), ''])
    (output / 'NANO_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(rows) * 0.9), 4))
    labels = [r['case'] for r in rows]
    axes[0].bar(labels, [r['processed_fps_wall'] for r in rows])
    axes[0].set(ylabel='Processed frames / wall second', title='Real sustained processing throughput')
    axes[1].bar(labels, [r['end_to_end']['p95_ms'] for r in rows])
    axes[1].set(ylabel='P95 end-to-end latency (ms)', title='Capture through saved overlay')
    for axis in axes:
        axis.tick_params(axis='x', labelrotation=45)
    fig.tight_layout()
    fig.savefig(output / 'nano_performance.png', dpi=180)
    plt.close(fig)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', default='results')
    parser.add_argument('--nano', nargs='*')
    args = parser.parse_args()
    summarize(args.runs, args.out)
    if args.nano:
        summarize_nano(args.nano, args.out)
