import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import zipfile

import requests

FILES = {'train': '1ZNTTbGjvPUHuLj3r3wEg8m12et7tDluz',
         'validation': '1I3BVEygFrCT3BmZL9byt2VFMt5ouCwRX',
         'test': '1A1mLgpOjioSQ1pzSUkU0Ce_3W_YozqgY'}


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.fields = {}
        self.action = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'form':
            self.action = attrs.get('action')
        if tag == 'input' and attrs.get('name'):
            self.fields[attrs['name']] = attrs.get('value', '')


def download(root, split):
    root.mkdir(parents=True, exist_ok=True)
    archive = root / (split + '.zip')
    record = root / (split + '.download.json')
    if record.exists() and archive.exists():
        receipt = json.loads(record.read_text())
        if receipt['bytes'] == archive.stat().st_size and (root / split).is_dir():
            return receipt
    url = 'https://drive.usercontent.google.com/download'
    session = requests.Session()
    response = session.get(url, params={'id': FILES[split], 'export': 'download'},
                           stream=True, timeout=(30, 120))
    response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type', ''):
        form = DownloadForm()
        page = response.text
        form.feed(page)
        response.close()
        if 'Virus scan warning' not in page or form.action != url:
            raise RuntimeError('Official download requires login/permission; do not save HTML as ZIP: ' + split)
        response = session.get(url, params=form.fields, stream=True, timeout=(30, 120))
        response.raise_for_status()
    if 'text/html' in response.headers.get('Content-Type', ''):
        raise RuntimeError('Download returned HTML, not the dataset: ' + split)
    part = archive.with_suffix('.zip.part')
    digest = hashlib.sha256()
    size = 0
    with part.open('wb') as output:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            if size == 0 and not chunk.startswith(b'PK\x03\x04'):
                raise RuntimeError('Response is not a ZIP archive')
            output.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    response.close()
    with zipfile.ZipFile(part) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError('ZIP CRC failed: ' + bad)
        resolved = root.resolve()
        for item in z.infolist():
            name = item.filename.replace('\\', '/')
            if name.startswith('/') or '..' in Path(name).parts or ':' in name:
                raise RuntimeError('Unsafe archive path: ' + name)
            target = (root / name).resolve()
            if not target.is_relative_to(resolved):
                raise RuntimeError('Archive path escaped destination')
        required = sum(i.file_size for i in z.infolist())
        if shutil.disk_usage(root).free < required + 1024 ** 3:
            raise RuntimeError('Insufficient free space to extract')
        z.extractall(root)
    part.replace(archive)
    receipt = {'split': split, 'source': 'https://drive.google.com/file/d/' + FILES[split] + '/view',
               'bytes': size, 'sha256': digest.hexdigest(), 'zip_crc_verified': True}
    record.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=list(FILES), default=list(FILES))
    args = parser.parse_args()
    for split in args.splits:
        print(json.dumps(download(args.root, split)), flush=True)
