import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from urllib.parse import urlsplit
import zipfile

import requests

from data import sha256
from learning import write_json

REPOSITORY = 'hogomex393-svg/Computer-Vision-cw2'
BASE = 'https://api.github.com/repos/' + REPOSITORY


class NoAuthentication(requests.auth.AuthBase):
    def __call__(self, request):
        request.headers.pop('Authorization', None)
        return request


def validate_bundle(archive):
    archive = Path(archive)
    recipe = json.loads(archive.with_suffix('.release.json').read_text(encoding='utf-8'))
    if not re.fullmatch(r'[0-9a-f]{40}', recipe['source_commit']) or recipe['selected'] not in ('S0', 'S1', 'S2', 'S3'):
        raise ValueError('Invalid source commit or selected student')
    if recipe['archive'] != archive.name or recipe['bytes'] != archive.stat().st_size or recipe['sha256'] != sha256(archive):
        raise ValueError('Release recipe does not match the archive')
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('manifest.json'))
        if manifest['source_commit'] != recipe['source_commit'] or manifest['selected'] != recipe['selected']:
            raise ValueError('Source/selection provenance differs')
        names = {entry['path'] for entry in manifest['files']}
        if len(names) != len(manifest['files']) or len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate archive/manifest paths are not allowed')
        experiments = ('T', 'S0', 'S1', 'S2', 'S3')
        result_files = ('run.json', 'epochs.jsonl', 'best_validation.json', 'test_clean.json',
                        'test_brightness.json', 'test_blur.json', 'test_jpeg.json')
        expected = {'nano.py', 'runtime.py', 'selection.json', 'ARTIFACTS.md'}
        expected.update('manifests/' + name for name in ('train.json', 'validation.json', 'test.json', 'audit.json'))
        expected.update('checkpoints/' + name + '.pt' for name in experiments)
        expected.update('results/' + name + '/' + filename for name in experiments for filename in result_files)
        if 'results/T/teacher_reuse.json' in names:
            expected.add('results/T/teacher_reuse.json')
        export_pairs = {(recipe['selected'], size) for size in (256, 384, 512)} | {('S0', 384)}
        expected.update('models/{}_{}{}'.format(name, size, suffix)
                        for name, size in export_pairs for suffix in ('.onnx', '.json', '.reference.npz'))
        if names != expected:
            raise ValueError('Archive does not contain exactly the expected project handoff files')
        if set(z.namelist()) != names | {'manifest.json'}:
            raise ValueError('Unexpected archive entries')
        for entry in manifest['files']:
            path = PurePosixPath(entry['path'])
            if path.is_absolute() or '..' in path.parts or ':' in str(path):
                raise ValueError('Unsafe archive path')
            if path.parts[0] not in ('checkpoints', 'models', 'results', 'manifests', 'nano.py', 'runtime.py', 'selection.json', 'ARTIFACTS.md'):
                raise ValueError('Archive contains material outside the project handoff')
            digest = hashlib.sha256()
            with z.open(entry['path']) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != entry['sha256'] or z.getinfo(entry['path']).file_size != entry['bytes']:
                raise ValueError('Artifact entry checksum mismatch: ' + entry['path'])
    return recipe


def github_session(account):
    if not re.fullmatch(r'[A-Za-z0-9-]{1,39}', account):
        raise ValueError('Invalid GitHub account name')
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never', GIT_TRACE='0', GCM_TRACE='0', GCM_TRACE_SECRETS='0')
    credential = subprocess.run(['git', '-c', 'credential.username=' + account, 'credential', 'fill'],
                                input='protocol=https\nhost=github.com\nusername=' + account + '\n\n',
                                capture_output=True, text=True, env=env, check=True)
    fields = dict(line.split('=', 1) for line in credential.stdout.splitlines() if '=' in line)
    token = fields.get('password')
    if not token:
        raise RuntimeError('No GitHub credential is available; use the official login flow')
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json',
                            'X-GitHub-Api-Version': '2026-03-10', 'User-Agent': 'cv-cw2-artifact-delivery'})
    fields.clear()
    identity = session.get('https://api.github.com/user', timeout=30)
    identity.raise_for_status()
    if identity.json()['login'].lower() != account.lower():
        raise RuntimeError('Authenticated account differs from the requested account')
    repository = session.get(BASE, timeout=30)
    repository.raise_for_status()
    if repository.json()['full_name'].lower() != REPOSITORY.lower() or not repository.json()['permissions'].get('push'):
        raise RuntimeError('The account cannot deliver to the designated repository')
    return session


def private_gate(release, asset):
    anonymous = requests.Session()
    anonymous.auth = NoAuthentication()
    anonymous.headers['Accept'] = 'application/vnd.github+json'
    if anonymous.get(BASE, timeout=30).status_code != 200:
        raise RuntimeError('Anonymous network control check failed')
    for url in (BASE + '/releases/' + str(release['id']), BASE + '/releases/assets/' + str(asset['id'])):
        if anonymous.get(url, allow_redirects=False, timeout=30).status_code not in (401, 404):
            raise RuntimeError('Draft material is accessible anonymously; refusing model upload')
    url = asset['browser_download_url']
    if urlsplit(url).hostname != 'github.com' or not urlsplit(url).path.startswith('/' + REPOSITORY + '/'):
        raise ValueError('Unexpected browser download host/path')
    response = anonymous.get(url, stream=True, allow_redirects=False, timeout=30)
    try:
        if response.status_code in (401, 404):
            return True
        location = urlsplit(response.headers.get('Location', ''))
        if response.status_code in (301, 302, 303, 307) and location.hostname in (None, 'github.com') and location.path.startswith('/login'):
            return True
        raise RuntimeError('Browser asset access is not proven private; refusing model upload')
    finally:
        response.close()


def upload_asset(session, release, path):
    path = Path(path)
    expected = sha256(path)
    response = session.get(BASE + '/releases/' + str(release['id']) + '/assets', params={'per_page': 100}, timeout=30)
    response.raise_for_status()
    existing = [a for a in response.json() if a['name'] == path.name]
    if existing:
        asset = existing[0]
    else:
        url = release['upload_url'].split('{', 1)[0]
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname != 'uploads.github.com' or parsed.path != '/repos/' + REPOSITORY + '/releases/' + str(release['id']) + '/assets':
            raise ValueError('Unexpected asset upload destination')
        with path.open('rb') as source:
            response = session.post(url, params={'name': path.name}, data=source, allow_redirects=False,
                                    headers={'Content-Type': 'application/zip' if path.suffix == '.zip' else 'application/json'},
                                    timeout=(30, 900))
        response.raise_for_status()
        if response.status_code != 201:
            raise RuntimeError('Upload did not return Created')
        asset = response.json()
    if asset['state'] != 'uploaded' or asset['size'] != path.stat().st_size:
        raise RuntimeError('Existing/uploaded asset differs; preserve it and use a new release version')
    if asset.get('digest'):
        if asset['digest'] != 'sha256:' + expected:
            raise RuntimeError('Server asset digest differs; remote verification failed')
    else:
        digest = hashlib.sha256()
        received = 0
        try:
            download = session.get(BASE + '/releases/assets/' + str(asset['id']),
                                   headers={'Accept': 'application/octet-stream'}, stream=True, timeout=(30, 120))
            with download:
                if download.status_code != 200:
                    raise RuntimeError('Authenticated asset read failed with HTTP ' + str(download.status_code))
                for chunk in download.iter_content(1024 * 1024):
                    digest.update(chunk)
                    received += len(chunk)
        except requests.RequestException:
            raise RuntimeError('Remote asset verification network error') from None
        if received != path.stat().st_size or digest.hexdigest() != expected:
            raise RuntimeError('Downloaded asset checksum differs')
    return asset


def publish(archive, account, tag):
    archive = Path(archive)
    recipe = validate_bundle(archive)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', tag):
        raise ValueError('Invalid release tag')
    session = github_session(account)
    commit = session.get(BASE + '/commits/' + recipe['source_commit'], timeout=30)
    commit.raise_for_status()
    if commit.json()['sha'] != recipe['source_commit']:
        raise ValueError('The package source commit is not available in the target repository')
    tag_response = session.get(BASE + '/git/ref/tags/' + tag, timeout=30)
    if tag_response.status_code == 200:
        obj = tag_response.json()['object']
        for _ in range(8):
            if obj['type'] != 'tag':
                break
            if not re.fullmatch(r'[0-9a-f]{40}', obj['sha']):
                raise ValueError('Invalid annotated tag object')
            resolved = session.get(BASE + '/git/tags/' + obj['sha'], timeout=30)
            resolved.raise_for_status()
            obj = resolved.json()['object']
        if obj['type'] != 'commit' or obj['sha'] != recipe['source_commit']:
            raise ValueError('The tag belongs to another source revision; preserve it and choose a new tag')
    elif tag_response.status_code != 404:
        tag_response.raise_for_status()
    releases = []
    for page in range(1, 11):
        response = session.get(BASE + '/releases', params={'per_page': 100, 'page': page}, timeout=30)
        response.raise_for_status()
        values = response.json()
        releases.extend(values)
        if len(values) < 100:
            break
    matches = [r for r in releases if r['tag_name'] == tag]
    marker = 'Artifact SHA256: ' + recipe['sha256']
    if matches:
        release = matches[0]
        if not release['draft'] or marker not in (release.get('body') or '') or release['target_commitish'] != recipe['source_commit']:
            raise ValueError('Existing release is not this private delivery draft; do not overwrite it')
    else:
        body = ('Host-validated CW2 models for repository collaborators.\n\n' + marker + '\n'
                'Source commit: ' + recipe['source_commit'] + '\n'
                'Selected student: ' + recipe['selected'] + '\n'
                'Original Jetson Nano hardware validation is still pending.\n'
                'Keep this release as a draft: publishing it makes the assets public.\n')
        response = session.post(BASE + '/releases', json={'tag_name': tag, 'target_commitish': recipe['source_commit'],
                                'name': 'CW2 host-validated models', 'body': body, 'draft': True,
                                'prerelease': True, 'make_latest': 'false'}, timeout=30)
        response.raise_for_status()
        release = response.json()
    if not release['draft']:
        raise RuntimeError('A non-draft release cannot be used for this private delivery')
    manifest_path = archive.with_suffix('.manifest.json')
    write_json(manifest_path, {k: recipe[k] for k in ('archive', 'bytes', 'sha256', 'source_commit', 'selected')})
    canary = upload_asset(session, release, manifest_path)
    private_gate(release, canary)
    asset = upload_asset(session, release, archive)
    private_gate(release, asset)
    record = {'repository': REPOSITORY, 'draft': True, 'release_url': release['html_url'],
              'asset_api_url': asset['url'], 'browser_download_url': asset['browser_download_url'],
              'bytes': asset['size'], 'sha256': recipe['sha256'], 'source_commit': recipe['source_commit'],
              'selected': recipe['selected'], 'anonymous_access_denied': True,
              'access': 'repository members with push permission; authenticate before downloading'}
    write_json(archive.with_suffix('.upload.json'), record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--account')
    parser.add_argument('--tag', default='cw2-host-v1')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        result = validate_bundle(args.archive)
    else:
        if not args.account:
            parser.error('--account is required for authenticated draft delivery')
        result = publish(args.archive, args.account, args.tag)
    print(json.dumps(result, indent=2))
