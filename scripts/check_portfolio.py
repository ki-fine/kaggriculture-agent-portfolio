"""Check public-file hashes, local Markdown links and path redaction."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'checksums.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f'Missing artifact: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Content hash changed: {name}')

    checked_links = 0
    link_pattern = re.compile(r'\]\(([^\s)]+)(?:\s+"[^"]*")?\)')
    private_path = re.compile(r'(?:[A-Za-z]:[\\/](?:Users|deepseekworkspace|Obsidian)|/home/[A-Za-z0-9_-]+/)')
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
            continue
        if path.suffix not in {'.md', '.json', '.py', '.yml', '.txt'}:
            continue
        text = path.read_text(encoding='utf-8')
        if private_path.search(text):
            raise ValueError(f'Private workstation path: {path.relative_to(ROOT)}')
        if path.suffix != '.md':
            continue
        for target in link_pattern.findall(text):
            if target.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            target = unquote(target.split('#', 1)[0])
            linked = (path.parent / target).resolve()
            if not linked.is_relative_to(ROOT) or not linked.exists():
                raise ValueError(f'Broken local link: {path.relative_to(ROOT)} -> {target}')
            checked_links += 1

    release = json.loads((ROOT / 'data/release_identity.json').read_text(encoding='utf-8'))
    assert release['kaggle']['submission_id'] == '56713467'
    assert release['kaggle']['status'] == 'accepted_by_platform'
    assert release['official_results']['status'] == 'pending'
    ml = json.loads((ROOT / 'data/ml_experiment.json').read_text(encoding='utf-8'))
    print(json.dumps({'status': 'PASS', 'hashed_files': len(manifest['files']),
                      'local_links': checked_links, 'ml_evidence_loaded': bool(ml)},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
