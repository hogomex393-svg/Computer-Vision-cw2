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
            protocols.add(json.dumps({key: run['config'][key] for key in (
                'size', 'batch_size', 'epochs', 'seed', 'augment', 'learning_rate', 'weight_decay', 'freeze_bn', 'decoder_channels')}, sort_keys=True))
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
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', default='results')
    args = parser.parse_args()
    summarize(args.runs, args.out)
